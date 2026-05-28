"""
ingestion/embedder.py
Semantic search wrapper over the ChromaDB vector store.

Usage:
    from ingestion.embedder import ContractSearcher
    searcher = ContractSearcher()
    results = searcher.search("penalty clauses for late delivery", top_k=5)
    for r in results:
        print(r["supplier_id"], r["score"], r["text"][:120])
"""

import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile

from utils.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    MIN_RELEVANCE_PCT,
)
from utils.logger import logger

# Works on Windows, macOS, and Linux
TMP_CHROMA_DIR = Path(tempfile.gettempdir()) / "chroma_store"


class ContractSearcher:
    """
    Lazy-loaded semantic search over procurement contracts stored in ChromaDB.

    On first use, copies the persisted ChromaDB from the workspace folder
    into /tmp so ChromaDB can open it with proper file locking.
    """

    def __init__(
        self,
        persist_dir: Path = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self._collection = None
        self._model = None

    def _load(self) -> None:
        """Initialise model + ChromaDB collection (called lazily on first search)."""
        import chromadb
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self._model = SentenceTransformer(self.embedding_model_name)

        # Copy workspace ChromaDB → /tmp (avoids virtiofs locking issues)
        if not self.persist_dir.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {self.persist_dir}. "
                "Run `python -m ingestion.document_ingester` first."
            )

        if TMP_CHROMA_DIR.exists():
            shutil.rmtree(TMP_CHROMA_DIR)
        shutil.copytree(self.persist_dir, TMP_CHROMA_DIR)
        logger.info(f"ChromaDB copied from workspace → /tmp for querying")

        client = chromadb.PersistentClient(path=str(TMP_CHROMA_DIR))
        self._collection = client.get_collection(self.collection_name)
        count = self._collection.count()
        logger.success(f"✅ ChromaDB ready — {count} chunks in '{self.collection_name}'")

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        supplier_filter: str | None = None,
        min_relevance_pct: float = MIN_RELEVANCE_PCT,
    ) -> list[dict[str, Any]]:
        """
        Semantic search over contract chunks.

        Args:
            query:             Natural language search query.
            top_k:             Number of results to return (before threshold filter).
            supplier_filter:   If provided, restrict results to this supplier_id.
            min_relevance_pct: Discard chunks with relevance below this value (0-100).
                               Prevents low-signal content from polluting LLM context.

        Returns:
            List of dicts with keys: supplier_id, source_file, chunk_index,
            section_title, text, score, relevance_pct.
            Ordered by relevance descending, filtered to >= min_relevance_pct.
        """
        if self._collection is None:
            self._load()

        query_embedding = self._model.encode(query).tolist()

        where = {"supplier_id": supplier_filter} if supplier_filter else None
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            relevance_pct = round((1 - dist) * 100, 1)
            if relevance_pct < min_relevance_pct:
                logger.debug(
                    f"Filtered chunk [{meta.get('source_file')} §{meta.get('section_title')}] "
                    f"— relevance {relevance_pct}% < threshold {min_relevance_pct}%"
                )
                continue
            results.append({
                "supplier_id": meta["supplier_id"],
                "source_file": meta["source_file"],
                "chunk_index": meta["chunk_index"],
                "section_title": meta.get("section_title", "Unknown"),
                "text": doc,
                "score": round(dist, 4),          # cosine distance (0=identical)
                "relevance_pct": relevance_pct,
            })

        return results

    def collection_stats(self) -> dict[str, Any]:
        """Return basic stats about the loaded collection."""
        if self._collection is None:
            self._load()
        return {
            "collection": self.collection_name,
            "total_chunks": self._collection.count(),
            "embedding_model": self.embedding_model_name,
            "persist_dir": str(self.persist_dir),
        }


# ── Convenience top-level function ───────────────────────────────────────────

_default_searcher: ContractSearcher | None = None


def search(query: str, top_k: int = TOP_K, supplier_filter: str | None = None) -> list[dict]:
    """Module-level shortcut: search(query) → results list."""
    global _default_searcher
    if _default_searcher is None:
        _default_searcher = ContractSearcher()
    return _default_searcher.search(query, top_k=top_k, supplier_filter=supplier_filter)


# ── CLI quick-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What are the penalty clauses for late delivery?",
        "payment terms and invoice due dates",
        "SLA compliance target and non-performance consequences",
        "contract termination notice period",
        "auto-renewal terms",
    ]

    searcher = ContractSearcher()
    print(f"\n{'='*70}")
    print(f"  ChromaDB Stats: {searcher.collection_stats()}")
    print(f"{'='*70}\n")

    for q in test_queries:
        print(f"\n🔍 Query: {q}")
        print("-" * 60)
        results = searcher.search(q, top_k=3)
        for r in results:
            section = r.get("section_title", "")
            section_str = f" § {section}" if section else ""
            print(f"  📄 {r['source_file']}{section_str}  [{r['relevance_pct']}% relevant]")
            print(f"     {r['text'][:200].replace(chr(10), ' ')}…")
            print()
            print()
