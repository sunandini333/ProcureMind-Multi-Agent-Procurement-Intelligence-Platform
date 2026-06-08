"""
pipeline.py
===========
RAGPipeline — the main orchestrator that wires together all modular components.

    Chunker  →  ContextualEnricher  →  VectorStore
                                               ↓
    User Query  →  Retriever  →  Reranker  →  LLM  →  Answer

Components can be configured via RAGConfig (string-based) or injected directly
as instances (for full control / custom subclasses).

Classes:
    RAGConfig    — dataclass of all pipeline options
    RAGPipeline  — main pipeline: index_document(), query(), compare_strategies()

Usage:
    pipeline = RAGPipeline(RAGConfig(
        chunker="hierarchical",
        use_contextual_retrieval=True,
        retriever="fusion",
        reranker="cross_encoder",
    ))
    pipeline.index_document("contract_2025", document_text)
    result = pipeline.query("What are the force majeure clauses?")
    print(result["answer"])
    print(result["sources"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic

from chunkers import BaseChunker, Chunk, FixedSizeChunker, HierarchicalChunker
from contextual import ContextualEnricher
from rerankers import BaseReranker, CrossEncoderReranker, NoOpReranker
from retrievers import BaseRetriever, RAGFusionRetriever, VectorRetriever
from vectorstore import ChromaVectorStore


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class RAGConfig:
    """
    All pipeline settings in one place.

    Chunker options:
        "hierarchical"  — structure-aware (recommended for procurement docs)
        "fixed_size"    — token-count baseline

    Retriever options:
        "vector"   — plain cosine similarity
        "fusion"   — RAG-Fusion with LLM query expansion + RRF

    Reranker options:
        "cross_encoder" — sentence-transformers cross-encoder (recommended)
        "none"          — no reranking (faster, weaker)
    """

    # ── Chunking ──────────────────────────────────────────────────
    chunker: str = "hierarchical"

    # ── Contextual Retrieval ──────────────────────────────────────
    use_contextual_retrieval: bool = True
    contextual_model: str = "claude-haiku-4-5-20251001"

    # ── Vector Store ──────────────────────────────────────────────
    collection_name: str = "procurement_rag"
    persist_directory: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Retrieval ─────────────────────────────────────────────────
    retriever: str = "vector"     # start simple; switch to "fusion" for better recall
    n_retrieve: int = 12          # candidates passed to reranker

    # ── Reranking ─────────────────────────────────────────────────
    reranker: str = "cross_encoder"
    n_final: int = 5              # chunks sent to the LLM

    # ── Answer Generation ─────────────────────────────────────────
    generation_model: str = "claude-sonnet-4-6"
    max_answer_tokens: int = 1024


# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a knowledgeable Supply Chain and Procurement assistant.

Answer questions using ONLY the document excerpts provided.
Rules:
  • Cite the section name whenever you use specific information from it.
  • If the excerpts don't fully answer the question, say so explicitly.
  • Be concise and precise — procurement professionals value clarity.
  • Flag any ambiguities or missing information that would affect a sourcing decision."""

_GENERATION_PROMPT = """\
Document excerpts (ranked by relevance):

{context}

---
Question: {question}

Answer:"""


# ── Pipeline ───────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline for Supply Chain / Procurement documents.

    Indexing:
        index_document(doc_id, text)
            → chunk  → (optionally enrich with contextual prefix)
            → embed  → store in ChromaDB

    Querying:
        query(question)
            → retrieve candidates  → rerank  → generate answer

    Swapping components:
        Pass a RAGConfig with different string values, or inject component
        instances directly via the constructor keyword arguments.

    Args:
        config             — RAGConfig instance (all defaults are sensible)
        chunker            — override with any BaseChunker instance
        retriever_override — override with any BaseRetriever instance
        reranker_override  — override with any BaseReranker instance
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        chunker: Optional[BaseChunker] = None,
        retriever_override: Optional[BaseRetriever] = None,
        reranker_override: Optional[BaseReranker] = None,
    ):
        self.config = config or RAGConfig()
        self._anthropic = call_llm

        # ── Vector Store (shared across components) ───────────────────────────
        self.vector_store = ChromaVectorStore(
            collection_name=self.config.collection_name,
            persist_directory=self.config.persist_directory,
            embedding_model=self.config.embedding_model,
        )

        # ── Chunker ───────────────────────────────────────────────────────────
        if chunker:
            self.chunker = chunker
        elif self.config.chunker == "hierarchical":
            self.chunker = HierarchicalChunker()
        else:
            self.chunker = FixedSizeChunker()

        # ── Contextual Enricher ───────────────────────────────────────────────
        self.enricher: Optional[ContextualEnricher] = (
            ContextualEnricher(model=self.config.contextual_model)
            if self.config.use_contextual_retrieval
            else None
        )

        # ── Retriever ─────────────────────────────────────────────────────────
        if retriever_override:
            self.retriever = retriever_override
        elif self.config.retriever == "fusion":
            self.retriever = RAGFusionRetriever(
                self.vector_store,
                n_results_per_query=self.config.n_retrieve,
            )
        else:
            self.retriever = VectorRetriever(self.vector_store)

        # ── Reranker ──────────────────────────────────────────────────────────
        if reranker_override:
            self.reranker = reranker_override
        elif self.config.reranker == "cross_encoder":
            self.reranker = CrossEncoderReranker()
        else:
            self.reranker = NoOpReranker()

        self._print_config()

    # ── Indexing ───────────────────────────────────────────────────────────────

    def index_document(
        self,
        doc_id: str,
        text: str,
        verbose: bool = True,
    ) -> List[Chunk]:
        """
        Full indexing pipeline for a single document.

        Steps:
            1. Chunk the document
            2. Enrich chunks with contextual prefixes (if enabled)
            3. Embed and store in the vector store

        Args:
            doc_id  — unique identifier for this document (used in citations)
            text    — full document text
            verbose — print step-by-step progress

        Returns:
            The list of Chunk objects that were indexed.
        """
        print(f"\n[Index] '{doc_id}'")

        # 1. Chunk
        if verbose:
            print(f"  Step 1/3  Chunking  ({self.chunker.__class__.__name__}) ...")
        chunks = self.chunker.chunk(text, doc_id)
        if verbose:
            levels = {}
            for c in chunks:
                levels[c.level] = levels.get(c.level, 0) + 1
            breakdown = ", ".join(f"{v} {k}" for k, v in levels.items())
            print(f"  → {len(chunks)} chunks  ({breakdown})")

        # 2. Contextual Enrichment
        if self.enricher:
            if verbose:
                print(f"  Step 2/3  Contextual enrichment  ({len(chunks)} API calls) ...")
            chunks = self.enricher.enrich(chunks, text, verbose=verbose)
        else:
            if verbose:
                print("  Step 2/3  Contextual enrichment  [skipped]")

        # 3. Store
        if verbose:
            print("  Step 3/3  Embedding + storing ...")
        self.vector_store.add_chunks(chunks)

        print(f"  ✓ Indexed {len(chunks)} chunks from '{doc_id}'\n")
        return chunks

    # ── Querying ───────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        n_results: Optional[int] = None,
        doc_filter: Optional[str] = None,
        return_sources: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Full query pipeline: retrieve → rerank → generate.

        Args:
            question      — the user's question
            n_results     — override for number of candidates to retrieve
            doc_filter    — restrict retrieval to a specific doc_id
            return_sources — include source excerpts in the result
            verbose       — print progress

        Returns:
            Dict with keys:
                answer   — the generated answer string
                question — echo of the input question
                sources  — list of source dicts (if return_sources=True)
                    Each source: chunk_id, section, breadcrumb, text_preview, score
        """
        n = n_results or self.config.n_retrieve
        where = {"doc_id": doc_filter} if doc_filter else None

        if verbose:
            print(f"\n[Query] '{question}'")

        # 1. Retrieve
        retrieved = self.retriever.retrieve(question, n_results=n, where=where)
        if verbose:
            print(f"  Retrieved  {len(retrieved)} candidates")

        # 2. Rerank
        final = self.reranker.rerank(question, retrieved, top_k=self.config.n_final)
        if verbose:
            print(f"  After rerank  {len(final)} chunks selected")

        # 3. Build context block
        context_parts: List[str] = []
        for i, chunk in enumerate(final):
            breadcrumb = chunk["metadata"].get("breadcrumb") or chunk["metadata"].get("section_title", "")
            header = f"[{i+1}] {breadcrumb}" if breadcrumb else f"[{i+1}]"
            context_parts.append(f"{header}\n{chunk['text']}")
        context = "\n\n---\n\n".join(context_parts)

        # 4. Generate
        response = self._anthropic.messages.create(
            model=self.config.generation_model,
            max_tokens=self.config.max_answer_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": _GENERATION_PROMPT.format(context=context, question=question),
            }],
        )
        answer = response.content[0].text.strip()

        result: Dict[str, Any] = {"answer": answer, "question": question}

        if return_sources:
            result["sources"] = [
                {
                    "chunk_id": c["chunk_id"],
                    "section": c["metadata"].get("section_title", ""),
                    "breadcrumb": c["metadata"].get("breadcrumb", ""),
                    "text_preview": (c["text"][:250] + " …") if len(c["text"]) > 250 else c["text"],
                    "score": round(c.get("rerank_score", c.get("rrf_score", c.get("score", 0.0))), 4),
                }
                for c in final
            ]

        return result

    # ── Strategy comparison ────────────────────────────────────────────────────

    def compare_strategies(
        self,
        question: str,
        strategies: List[Dict[str, Any]],
    ) -> None:
        """
        Run the same question through multiple pipeline configurations and
        print a side-by-side comparison.  Useful for picking the best strategy
        for your document type.

        Each strategy is a dict:
            {
                "label":    "Hierarchical + CrossEncoder",
                "chunker":  "hierarchical",
                "retriever": "vector",
                "reranker":  "cross_encoder",
                ... (any RAGConfig field)
            }

        NOTE: All strategies share the SAME vector store — so you should
        index documents before calling this method.

        Example:
            pipeline.compare_strategies("What is the payment term?", [
                {"label": "Baseline",  "chunker": "fixed_size",    "reranker": "none"},
                {"label": "Full stack","chunker": "hierarchical",  "reranker": "cross_encoder"},
            ])
        """
        sep = "=" * 65
        print(f"\n{sep}\nCOMPARING STRATEGIES\nQuestion: {question!r}\n{sep}")

        for strat in strategies:
            label = strat.pop("label", "unnamed")
            cfg = RAGConfig(**{**vars(self.config), **strat})
            print(f"\n▶ {label}")

            tmp = RAGPipeline(
                config=cfg,
                # Reuse the same vector store — no re-indexing needed
                retriever_override=RAGFusionRetriever(self.vector_store) if cfg.retriever == "fusion"
                    else VectorRetriever(self.vector_store),
                reranker_override=CrossEncoderReranker() if cfg.reranker == "cross_encoder"
                    else NoOpReranker(),
            )
            result = tmp.query(question, verbose=False, return_sources=True)
            print(f"  Answer : {result['answer'][:400]} …")
            sections = [s["section"] for s in result.get("sources", [])]
            print(f"  Sources: {sections}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _print_config(self) -> None:
        print("\n┌─ RAGPipeline ─────────────────────────────────────────")
        print(f"│  Chunker              : {self.chunker.__class__.__name__}")
        print(f"│  Contextual Retrieval : {'✓ enabled' if self.enricher else '✗ disabled'}")
        print(f"│  Retriever            : {self.retriever.__class__.__name__}")
        print(f"│  Reranker             : {self.reranker.__class__.__name__}")
        print(f"│  Generation Model     : {self.config.generation_model}")
        print(f"│  Vector Store         : {self.vector_store}")
        print("└──────────────────────────────────────────────────────\n")
