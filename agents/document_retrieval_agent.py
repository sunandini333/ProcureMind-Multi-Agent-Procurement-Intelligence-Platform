"""
agents/document_retrieval_agent.py
Semantic search over contracts → Claude synthesises an answer from retrieved chunks.

Flow:
  1. Embed the user's question with all-MiniLM-L6-v2.
  2. Retrieve top-k contract chunks from ChromaDB.
  3. Claude reads the chunks and answers the question in prose.
  4. Returns AgentResponse with answer, source chunks, and contract citations.

Run directly:
  python -m agents.document_retrieval_agent
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.llm_client import call_llm
from ingestion.embedder import ContractSearcher
from utils.config import TOP_K, MIN_RELEVANCE_PCT
from utils.logger import logger


# ── System prompt ─────────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM = """You are a procurement contract analyst. You will be given:
  1. A user's question about procurement contracts.
  2. Relevant excerpts retrieved from supplier contracts.

Your job:
- Answer the question directly and accurately using ONLY the provided excerpts.
- If multiple contracts are relevant, synthesise across them (note differences if significant).
- Quote specific clause numbers or figures where helpful (e.g. "Section 3.3 states…").
- If the excerpts do not contain enough information, say so clearly.
- Keep the answer concise (3-5 sentences) but complete.
- Do NOT invent information not present in the excerpts.
- Mention which suppliers' contracts the information comes from.
"""


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    question: str
    answer: str
    chunks_used: list[dict[str, Any]]
    citations: list[str] = field(default_factory=list)
    error: str | None = None

    def display(self) -> None:
        print(f"\n{'─'*65}")
        print(f"❓ {self.question}")
        print(f"{'─'*65}")
        if self.error:
            print(f"⚠️  Error: {self.error}")
        else:
            print(f"\n💬 {self.answer}")
            print(f"\n📄 Sources ({len(self.chunks_used)} chunks):")
            for c in self.chunks_used:
                section = c.get("section_title", "")
                section_str = f" § {section}" if section else ""
                print(f"   {c['source_file']}{section_str}  [{c['relevance_pct']}%]  — {c['text'][:100]}…")
            if self.citations:
                print(f"\n📎 Citations: {', '.join(self.citations)}")
        print()


# ── Agent class ───────────────────────────────────────────────────────────────

class DocumentRetrievalAgent:
    """
    Answers natural language questions about procurement contract terms
    by searching ChromaDB and synthesising with Claude.
    """

    def __init__(self, top_k: int = TOP_K, min_relevance_pct: float = MIN_RELEVANCE_PCT):
        self.top_k = top_k
        self.min_relevance_pct = min_relevance_pct
        self._searcher: ContractSearcher | None = None

    @property
    def searcher(self) -> ContractSearcher:
        if self._searcher is None:
            self._searcher = ContractSearcher()
        return self._searcher

    def run(
        self,
        question: str,
        supplier_filter: str | None = None,
    ) -> AgentResponse:
        """
        Full pipeline: question → retrieve chunks → synthesise → AgentResponse.

        Args:
            question:        Natural language question about contract terms.
            supplier_filter: Optional supplier_id to restrict search scope.
        """
        logger.info(f"DocumentRetrievalAgent: '{question[:80]}'")

        # Step 1 — Retrieve (threshold filtering is applied inside ContractSearcher.search)
        try:
            chunks = self.searcher.search(
                question,
                top_k=self.top_k,
                supplier_filter=supplier_filter,
                min_relevance_pct=self.min_relevance_pct,
            )
        except FileNotFoundError as e:
            return AgentResponse(
                question=question,
                answer=str(e),
                chunks_used=[],
                error=str(e),
            )

        if not chunks:
            return AgentResponse(
                question=question,
                answer=(
                    "No contract sections with sufficient relevance were found for this question. "
                    f"(Threshold: {self.min_relevance_pct}% — try rephrasing or broadening your query.)"
                ),
                chunks_used=[],
            )

        # Step 2 — Build context block for Claude (include section title for localisation)
        context_parts = []
        for i, c in enumerate(chunks, 1):
            section = c.get("section_title", "Unknown Section")
            context_parts.append(
                f"[Excerpt {i} — {c['source_file']} § {section} (relevance {c['relevance_pct']}%)]\n"
                f"{c['text']}"
            )
        context = "\n\n".join(context_parts)

        user_msg = f"Question: {question}\n\nRelevant contract excerpts:\n\n{context}"

        # Step 3 — Synthesise
        answer = call_llm(
            system=SYNTHESIS_SYSTEM,
            user=user_msg,
            max_tokens=700,
        )

        # Step 4 — Citations
        citations = [
            f"contract_{c['supplier_id']}"
            for c in chunks
        ]
        # Deduplicate while preserving order
        seen, unique_citations = set(), []
        for cit in citations:
            if cit not in seen:
                seen.add(cit)
                unique_citations.append(cit)

        return AgentResponse(
            question=question,
            answer=answer,
            chunks_used=chunks,
            citations=unique_citations,
        )


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = DocumentRetrievalAgent()

    demo_questions = [
        "What are the penalty clauses for late delivery?",
        "What notice period is required to terminate a contract?",
        "What happens if a supplier misses the SLA target for 3 months in a row?",
        "How does the auto-renewal clause work?",
        "What are the payment terms and late payment penalties?",
    ]

    for q in demo_questions:
        result = agent.run(q)
        result.display()
