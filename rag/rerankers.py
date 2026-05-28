"""
rerankers.py
============
Post-retrieval reranking strategies.

After the retriever returns its top-K chunks, a reranker re-scores the set
with a more powerful (but slower) model.  The cross-encoder jointly processes
(query, chunk) pairs — unlike embedding similarity which encodes query and
chunk independently — giving more accurate relevance scores.

Classes:
    BaseReranker          — abstract interface
    NoOpReranker          — pass-through (useful as baseline in A/B tests)
    CrossEncoderReranker  — uses a sentence-transformers cross-encoder model

Usage:
    reranker = CrossEncoderReranker()
    top5     = reranker.rerank("payment terms", retrieved_chunks, top_k=5)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


# ── Abstract base ──────────────────────────────────────────────────────────────

class BaseReranker(ABC):
    """
    Common interface for reranking strategies.

    Input:  a query string + list of result dicts from a retriever
    Output: subset of those results, re-ranked and trimmed to top_k
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank *results* against *query* and return the top *top_k*.

        Args:
            query   — the original user question
            results — list of result dicts from a retriever
            top_k   — how many results to keep

        Returns:
            Reranked and truncated list.  Each result gets a ``rerank_score`` key.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ── No-op reranker ─────────────────────────────────────────────────────────────

class NoOpReranker(BaseReranker):
    """
    Passes results through unchanged.

    Use this as a baseline when you want to compare retrieval-only
    performance against retrieval + reranking.  Also handy when
    sentence-transformers is not installed.
    """

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        trimmed = results[:top_k]
        for r in trimmed:
            r.setdefault("rerank_score", r.get("rrf_score", r.get("score", 0.0)))
        return trimmed


# ── Cross-encoder reranker ─────────────────────────────────────────────────────

class CrossEncoderReranker(BaseReranker):
    """
    Reranks chunks using a sentence-transformers CrossEncoder model.

    A cross-encoder processes the (query, chunk) pair *jointly*, meaning the
    query tokens directly attend to the chunk tokens.  This is substantially
    more accurate than independent bi-encoder (embedding) similarity, at the
    cost of higher latency.  Typical pattern: retrieve 20–50 candidates with
    a fast bi-encoder, then rerank to top 5 with a cross-encoder.

    Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
        - Fast (small), strong on passage-level relevance
        - Good trade-off for supply chain docs (contracts, specs, SOPs)

    Alternatives:
        - ``cross-encoder/ms-marco-MiniLM-L-12-v2``  (larger, ~10% better)
        - ``cross-encoder/ms-marco-electra-base``     (best quality, slowest)

    Args:
        model_name — HuggingFace cross-encoder model (default: MiniLM-L-6-v2)

    Falls back to NoOpReranker behaviour if sentence-transformers is not installed.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model_name = model_name
        self._model = None
        self._available = False

        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._model = CrossEncoder(model_name)
            self._available = True
        except ImportError:
            print(
                "⚠  sentence-transformers not installed — CrossEncoderReranker "
                "will fall back to score-passthrough.\n"
                "   Install with: pip install sentence-transformers"
            )
        except Exception as exc:
            print(f"⚠  CrossEncoder load failed ({exc}). Using score-passthrough.")

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not self._available or not results:
            # Graceful degradation: sort by existing score and trim
            for r in results:
                r.setdefault("rerank_score", r.get("rrf_score", r.get("score", 0.0)))
            return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, r["text"]) for r in results]
        scores = self._model.predict(pairs)   # returns numpy array

        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def __repr__(self) -> str:
        status = "loaded" if self._available else "unavailable"
        return f"CrossEncoderReranker(model={self.model_name!r}, status={status})"
