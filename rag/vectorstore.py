"""
vectorstore.py
==============
ChromaDB-backed vector store for RAG chunks.

Sentence-Transformer embeddings are used by default (runs locally, no API key).
The store persists to disk so indexed documents survive between sessions.

Classes:
    ChromaVectorStore — wraps a single ChromaDB collection

Usage:
    store = ChromaVectorStore(collection_name="procurement_v1")
    store.add_chunks(chunks)
    results = store.query("payment terms net-30", n_results=10)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

from chunkers import Chunk


class ChromaVectorStore:
    """
    Persistent ChromaDB collection storing embedded chunks.

    Storage layout per chunk:
        id       — chunk.chunk_id (used for upsert deduplication)
        document — chunk.content_for_embedding  (contextual prefix + text)
        metadata — original_text, doc_id, level, section_title, breadcrumb, ...

    The ``original_text`` metadata key preserves the raw chunk text so retrieval
    results can surface the human-readable excerpt, not the enriched embedding text.

    Args:
        collection_name  — name of the ChromaDB collection (default: 'procurement_rag')
        persist_directory — path to persist the DB; None → in-memory only
        embedding_model  — sentence-transformers model name
    """

    def __init__(
        self,
        collection_name: str = "procurement_rag",
        persist_directory: Optional[str] = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # ── Client ────────────────────────────────────────────────────────────
        if persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_directory))
        else:
            self._client = chromadb.EphemeralClient()

        # ── Embedding function (local, no API key required) ───────────────────
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # ── Collection ────────────────────────────────────────────────────────
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Embed and store *chunks*.  Uses upsert so re-indexing the same doc_id
        is safe — existing chunks are replaced, not duplicated.
        """
        if not chunks:
            return

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.content_for_embedding)
            metadatas.append({
                "doc_id": chunk.doc_id,
                "level": chunk.level,
                "section_title": chunk.section_title or "",
                "parent_section": chunk.parent_section or "",
                "original_text": chunk.text,
                "has_context_prefix": bool(chunk.contextual_prefix),
                # Flatten extra metadata — ChromaDB only accepts str/int/float/bool
                **{
                    k: str(v)
                    for k, v in chunk.metadata.items()
                    if isinstance(v, (str, int, float, bool))
                },
            })

        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed *query_text* and return the top *n_results* nearest chunks.

        Each result dict contains:
            chunk_id      — unique chunk identifier
            text          — original (non-prefixed) chunk text
            embedded_text — the text that was actually embedded
            score         — cosine similarity score in [0, 1]
            metadata      — full metadata dict from ChromaDB

        Args:
            query_text — the user's question or search string
            n_results  — how many chunks to retrieve
            where      — optional ChromaDB metadata filter, e.g. {"doc_id": "contract_a"}
        """
        kwargs: Dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(n_results, self.count or 1),
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        results: List[Dict[str, Any]] = []
        for chunk_id, document, meta, distance in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            results.append({
                "chunk_id": chunk_id,
                "text": meta.get("original_text", document),
                "embedded_text": document,
                "score": round(1.0 - distance, 4),   # cosine distance → similarity
                "metadata": meta,
            })

        return results

    # ── Utility ────────────────────────────────────────────────────────────────

    def delete_doc(self, doc_id: str) -> None:
        """Remove all chunks belonging to *doc_id* from the collection."""
        self._collection.delete(where={"doc_id": doc_id})

    def reset(self) -> None:
        """Drop and recreate the collection (clears all data)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        """Total number of chunks currently stored."""
        return self._collection.count()

    def __repr__(self) -> str:
        return (
            f"ChromaVectorStore(collection={self.collection_name!r}, "
            f"chunks={self.count}, model={self.embedding_model!r})"
        )
