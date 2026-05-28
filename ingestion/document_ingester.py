"""
ingestion/document_ingester.py
Reads all contract .txt files → chunks them → embeds with sentence-transformers
→ persists in ChromaDB.

ChromaDB is built in /tmp first (virtiofs doesn't support the file-locking
ChromaDB needs), then synced to the workspace vector_store/ directory.

Run: python -m ingestion.document_ingester
"""

import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile

from utils.config import (
    CONTRACTS_DIR,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from utils.logger import logger

# Works on Windows, macOS, and Linux
TMP_CHROMA_DIR = Path(tempfile.gettempdir()) / "chroma_store"


# ── Section header detection ──────────────────────────────────────────────────

# Matches patterns like: "3.", "3.1", "3.1.2", optionally followed by a title word
_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+([A-Z][^\n]{0,80})$")


def _extract_section_title(line: str) -> str | None:
    """Return a normalised section title if the line looks like a contract heading."""
    m = _SECTION_RE.match(line.strip())
    if m:
        return f"{m.group(1)} {m.group(2).strip()}"
    # Also catch plain ALL-CAPS headings like "PAYMENT TERMS"
    stripped = line.strip()
    if stripped.isupper() and 4 <= len(stripped) <= 80 and " " in stripped:
        return stripped.title()
    return None


# ── Text chunking ─────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Section-aware sliding-window chunker for contract documents.

    Strategy:
      1. Walk lines and detect numbered section headings (e.g. "3.1 Payment Terms").
      2. Track the current section title so every chunk knows which clause it belongs to.
      3. Merge lines into chunks of ~chunk_size characters with overlap at boundaries.

    Returns:
        List of dicts: {"text": str, "section_title": str}
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chunks: list[dict] = []
    current: list[str] = []
    current_len = 0
    current_section = "Preamble"

    def _flush(buf: list[str], section: str) -> None:
        if buf:
            chunks.append({"text": " ".join(buf), "section_title": section})

    for line in lines:
        detected = _extract_section_title(line)
        if detected:
            # New section heading — flush current buffer first
            _flush(current, current_section)
            current_section = detected
            # Start new buffer with the heading itself so it stays with its content
            current = [line]
            current_len = len(line)
            continue

        line_len = len(line)
        if current_len + line_len > chunk_size and current:
            _flush(current, current_section)
            # Overlap: keep last `overlap` chars worth of lines
            overlap_buf: list[str] = []
            overlap_len = 0
            for prev in reversed(current):
                if overlap_len + len(prev) <= overlap:
                    overlap_buf.insert(0, prev)
                    overlap_len += len(prev)
                else:
                    break
            current, current_len = overlap_buf, overlap_len

        current.append(line)
        current_len += line_len

    _flush(current, current_section)
    return chunks


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_contracts(
    contracts_dir: Path = CONTRACTS_DIR,
    chroma_dir: Path = TMP_CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
) -> int:
    """
    Load all contract .txt files, chunk, embed, and store in ChromaDB.
    Returns total number of chunks stored.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    contract_files = sorted(contracts_dir.glob("contract_SUP*.txt"))
    if not contract_files:
        logger.error(f"No contract files found in {contracts_dir}")
        return 0

    logger.info(f"Found {len(contract_files)} contract files")

    # Load embedding model
    logger.info(f"Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)
    logger.success("Embedding model loaded")

    # Set up ChromaDB in /tmp
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    all_docs, all_embeddings, all_metadatas, all_ids = [], [], [], []

    for contract_path in contract_files:
        supplier_id = contract_path.stem.replace("contract_", "")   # e.g. SUP0001
        text = contract_path.read_text(encoding="utf-8")
        chunk_dicts = chunk_text(text)

        logger.info(f"  {supplier_id}: {len(chunk_dicts)} chunks")

        for idx, chunk_dict in enumerate(chunk_dicts):
            chunk_id = f"{supplier_id}_chunk_{idx:03d}"
            chunk_text_val = chunk_dict["text"]
            embedding = model.encode(chunk_text_val).tolist()

            all_ids.append(chunk_id)
            all_docs.append(chunk_text_val)
            all_embeddings.append(embedding)
            all_metadatas.append({
                "supplier_id": supplier_id,
                "source_file": contract_path.name,
                "chunk_index": idx,
                "total_chunks": len(chunk_dicts),
                "section_title": chunk_dict["section_title"],
            })
            total_chunks += 1

    # Batch upsert
    logger.info(f"Upserting {total_chunks} chunks into ChromaDB collection '{collection_name}'…")
    batch_size = 100
    for i in range(0, total_chunks, batch_size):
        collection.upsert(
            ids=all_ids[i:i+batch_size],
            documents=all_docs[i:i+batch_size],
            embeddings=all_embeddings[i:i+batch_size],
            metadatas=all_metadatas[i:i+batch_size],
        )

    logger.success(f"✅ Stored {total_chunks} chunks across {len(contract_files)} contracts")
    return total_chunks


def sync_to_workspace(
    tmp_dir: Path = TMP_CHROMA_DIR,
    dest_dir: Path = CHROMA_PERSIST_DIR,
) -> None:
    """
    Copy the ChromaDB from /tmp into the persistent workspace folder.
    Uses file-by-file copy to work around virtiofs rmtree restrictions.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    for src_file in tmp_dir.rglob("*"):
        if not src_file.is_file():
            continue
        relative = src_file.relative_to(tmp_dir)
        dest_file = dest_dir / relative
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)

    size_mb = sum(f.stat().st_size for f in dest_dir.rglob("*") if f.is_file()) / 1_048_576
    logger.success(f"✅ ChromaDB synced to workspace: {dest_dir} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    logger.info("📄 Starting document ingestion…")
    total = ingest_contracts()
    if total > 0:
        sync_to_workspace()
    logger.info("✅ Document ingestion complete.")
