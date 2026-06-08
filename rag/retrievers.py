"""
retrievers.py
=============
Retrieval strategies for the RAG pipeline.

Classes:
    BaseRetriever       — abstract interface; all retrievers implement .retrieve()
    VectorRetriever     — plain cosine-similarity retrieval from the vector store
    RAGFusionRetriever  — generates N query variants via LLM, retrieves for each,
                          then merges with Reciprocal Rank Fusion (RRF)

Stand-alone helper:
    reciprocal_rank_fusion(result_lists, k) → merged list

Usage:
    store     = ChromaVectorStore(...)
    retriever = RAGFusionRetriever(store, n_query_variants=3)
    results   = retriever.retrieve("What are the delivery lead times?", n_results=8)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

import anthropic

from vectorstore import ChromaVectorStore


# ── Abstract base ──────────────────────────────────────────────────────────────

class BaseRetriever(ABC):
    """
    Common interface for all retrieval strategies.

    Every retriever returns a list of result dicts with at least:
        chunk_id, text, score, metadata
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top *n_results* chunks relevant to *query*.

        Args:
            query     — natural-language question or search string
            n_results — number of chunks to return
            where     — optional metadata filter passed to the vector store

        Returns:
            List of result dicts, ranked by relevance (most relevant first).
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ── Plain vector retriever ─────────────────────────────────────────────────────

class VectorRetriever(BaseRetriever):
    """
    Retrieves chunks by cosine similarity of the query embedding.

    This is the simplest and fastest strategy — a good baseline.
    Weakness: sensitive to exact phrasing; misses synonyms and paraphrases.

    Args:
        vector_store — ChromaVectorStore instance to query
    """

    def __init__(self, vector_store: ChromaVectorStore):
        self.store = vector_store

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return self.store.query(query, n_results=n_results, where=where)


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score for chunk c across lists:  Σ  1 / (k + rank(c, list_i))

    k=60 is the standard constant from the original RRF paper — it dampens
    the influence of very high ranks without ignoring lower-ranked results.

    Args:
        result_lists — list of ranked result lists (each list from one query)
        k            — RRF damping constant (default 60)

    Returns:
        Single merged list sorted by descending RRF score.
        Each result carries an added ``rrf_score`` key.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    chunk_data: Dict[str, Dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            cid = result["chunk_id"]
            rrf_scores[cid] += 1.0 / (k + rank + 1)
            chunk_data[cid] = result   # last write wins for metadata

    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

    merged: List[Dict[str, Any]] = []
    for cid in sorted_ids:
        item = {**chunk_data[cid], "rrf_score": round(rrf_scores[cid], 6)}
        merged.append(item)

    return merged


# ── RAG-Fusion retriever ───────────────────────────────────────────────────────

_QUERY_EXPANSION_PROMPT = """\
You are a Supply Chain and Procurement domain expert.

Given the search query below, generate {n} alternative phrasings that capture \
the same information need from different angles.

Focus on:
  - Procurement / supply chain synonyms (e.g. "lead time" ↔ "delivery schedule")
  - Related data points someone might search for
  - Both formal and informal terminology

Original query: {query}

Return ONLY the alternative queries, one per line, no numbering or extra text."""


class RAGFusionRetriever(BaseRetriever):
    """
    RAG-Fusion: expands the query into N variants, retrieves for each,
    then merges and re-ranks results with Reciprocal Rank Fusion.

    Why this helps:
      • Single-query retrieval is brittle — a slightly different phrasing
        of the same question can retrieve completely different chunks.
      • Multiple query variants collectively cast a wider net, and RRF
        promotes chunks that appear highly ranked across several variants.

    Args:
        vector_store        — ChromaVectorStore instance
        n_query_variants    — number of LLM-generated alternative queries (default 3)
        n_results_per_query — chunks retrieved per query variant (default 8)
        model               — Anthropic model for query expansion (default: Haiku)
        rrf_k               — RRF damping constant (default 60)
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        n_query_variants: int = 3,
        n_results_per_query: int = 8,
        model: str = "claude-haiku-4-5-20251001",
        rrf_k: int = 60,
    ):
        self.store = vector_store
        self.n_query_variants = n_query_variants
        self.n_results_per_query = n_results_per_query
        self.rrf_k = rrf_k
        self._client = call_llm
        self._model = model

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        1. Expand *query* into N variants.
        2. Retrieve *n_results_per_query* chunks for each variant.
        3. Merge all result lists with RRF.
        4. Return the top *n_results* fused results.
        """
        variants = self._expand_query(query)
        print(f"  [RAGFusion] Query variants ({len(variants)}):")
        for v in variants:
            print(f"    • {v}")

        all_lists: List[List[Dict[str, Any]]] = []
        for variant in variants:
            results = self.store.query(
                variant,
                n_results=self.n_results_per_query,
                where=where,
            )
            all_lists.append(results)

        merged = reciprocal_rank_fusion(all_lists, k=self.rrf_k)
        return merged[:n_results]

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _expand_query(self, query: str) -> List[str]:
        """Return [original_query] + LLM-generated variants."""
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": _QUERY_EXPANSION_PROMPT.format(
                        n=self.n_query_variants,
                        query=query,
                    ),
                }],
            )
            raw = response.content[0].text.strip()
            variants = [line.strip() for line in raw.splitlines() if line.strip()]
            return [query] + variants[: self.n_query_variants]
        except Exception as exc:
            print(f"  [RAGFusion] Query expansion failed ({exc}); using original query only.")
            return [query]
