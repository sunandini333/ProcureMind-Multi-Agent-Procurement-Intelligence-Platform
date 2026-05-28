"""
agents/orchestrator.py
Master Procurement CoPilot agent — routes questions, runs sub-agents,
synthesises hybrid answers, and maintains conversation memory.

Architecture
────────────
  User question
       │
       ▼
  IntentClassifier  (Claude, fast)
       │
       ├─── "structured"  ──► StructuredQueryAgent   (NL→SQL)
       ├─── "document"    ──► DocumentRetrievalAgent  (ChromaDB RAG)
       ├─── "hybrid"      ──► Both agents → HybridSynthesizer
       └─── "out_of_scope"──► Polite deflection

  Conversation history (last N turns) is injected into every prompt
  so follow-up questions resolve correctly.

Run directly for a CLI demo:
  python -m agents.orchestrator
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.llm_client import call_llm
from agents.structured_query_agent import StructuredQueryAgent
from agents.document_retrieval_agent import DocumentRetrievalAgent
from utils.logger import logger


# ── Types ─────────────────────────────────────────────────────────────────────

Intent = Literal["structured", "document", "hybrid", "out_of_scope"]

MAX_HISTORY_TURNS = 6   # keep last 6 Q&A pairs in context


# ── Prompts ───────────────────────────────────────────────────────────────────

INTENT_SYSTEM = """You are a routing agent for a Procurement CoPilot system.

Your job is to classify the user's question into exactly ONE of these categories:

  structured   — needs data from the structured database (spend figures, SLA %,
                 PO counts, supplier lists, renewal dates, price spikes, invoice
                 matching, supplier rankings, numeric metrics).

  document     — needs information from contract text (penalty clauses, payment
                 terms, termination conditions, auto-renewal clauses, SLA
                 definitions, force majeure, compliance requirements, legal text).

  hybrid       — needs BOTH: e.g. "Which supplier has the worst SLA and what
                 does their contract say about penalties?"

  out_of_scope — completely unrelated to procurement (weather, coding, cooking, etc.)

Reply with ONLY one word: structured, document, hybrid, or out_of_scope.
"""

HYBRID_SYNTHESIS_SYSTEM = """You are a senior procurement analyst.

You will receive:
1. A user question.
2. A structured data answer (from the procurement database — spend, SLA, POs).
3. A document answer (from contract clause analysis).

Your job: synthesise ONE coherent, executive-ready answer that combines both sources.
- Lead with the most important finding.
- Connect the data insight to the relevant contract clause.
- Flag any risks or recommended actions.
- Keep it to 4-6 sentences max.
- Do NOT mention internal system names (databases, ChromaDB, SQL, agents).
"""

OUT_OF_SCOPE_MSG = (
    "I'm your Procurement CoPilot — I can help with supplier spend analysis, "
    "contract terms, SLA performance, purchase orders, and renewal risk. "
    "That question falls outside my scope. Try asking something like: "
    "'Which suppliers are up for renewal this quarter?' or "
    "'What do our contracts say about late-delivery penalties?'"
)


# ── OrchestratorResponse ──────────────────────────────────────────────────────

@dataclass
class OrchestratorResponse:
    """Unified response object returned to the UI."""
    question: str
    answer: str
    intent: Intent
    # Structured agent results (if used)
    sql_used: str | None = None
    structured_rows: list[dict] | None = None
    # Document agent results (if used)
    contract_chunks: list[dict] | None = None
    # Merged citations
    citations: list[str] = field(default_factory=list)
    error: str | None = None

    def display(self) -> None:
        """Pretty-print to terminal."""
        bar = "─" * 65
        print(f"\n{bar}")
        print(f"❓ {self.question}")
        print(f"🎯 Intent: {self.intent.upper()}")
        print(bar)
        if self.error:
            print(f"⚠️  {self.error}")
        else:
            print(f"\n💬 {self.answer}")
        if self.sql_used:
            print(f"\n🗄️  SQL: {self.sql_used}")
        if self.structured_rows:
            print(f"📊 Rows: {len(self.structured_rows)}")
        if self.contract_chunks:
            print(f"📄 Chunks: {len(self.contract_chunks)}")
        if self.citations:
            print(f"📎 Citations: {', '.join(self.citations)}")
        print()


# ── History helper ────────────────────────────────────────────────────────────

def _format_history(history: list[dict]) -> str:
    """Convert history list to a readable block for prompts."""
    if not history:
        return ""
    lines = ["Previous conversation (most recent last):"]
    for turn in history[-MAX_HISTORY_TURNS:]:
        lines.append(f"  User: {turn['question']}")
        lines.append(f"  Assistant: {turn['answer'][:300]}{'…' if len(turn['answer']) > 300 else ''}")
    return "\n".join(lines)


# ── Orchestrator ──────────────────────────────────────────────────────────────

class ProcurementOrchestrator:
    """
    Top-level agent that routes questions to sub-agents and synthesises
    combined answers. Maintains rolling conversation history.
    """

    def __init__(self) -> None:
        self._structured_agent: StructuredQueryAgent | None = None
        self._document_agent: DocumentRetrievalAgent | None = None
        self.history: list[dict] = []   # list of {question, answer, intent}

    # ── Lazy-loaded sub-agents ────────────────────────────────────────────────

    @property
    def structured_agent(self) -> StructuredQueryAgent:
        if self._structured_agent is None:
            self._structured_agent = StructuredQueryAgent()
        return self._structured_agent

    @property
    def document_agent(self) -> DocumentRetrievalAgent:
        if self._document_agent is None:
            self._document_agent = DocumentRetrievalAgent()
        return self._document_agent

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, question: str) -> OrchestratorResponse:
        """
        Process a user question end-to-end and return an OrchestratorResponse.
        Automatically updates conversation history.
        """
        logger.info(f"Orchestrator ← '{question[:80]}'")

        # 1 — Resolve coreferences using history context
        resolved_question = self._resolve_question(question)

        # 2 — Classify intent
        intent = self._classify_intent(resolved_question)
        logger.info(f"Intent → {intent}")

        # 3 — Route
        response = self._route(resolved_question, intent)

        # 4 — Store in history
        self.history.append({
            "question": question,
            "answer": response.answer,
            "intent": intent,
        })

        return response

    def clear_history(self) -> None:
        """Reset conversation memory (new session)."""
        self.history.clear()
        logger.info("Conversation history cleared.")

    # ── Intent classification ─────────────────────────────────────────────────

    def _classify_intent(self, question: str) -> Intent:
        history_block = _format_history(self.history)
        user_msg = question
        if history_block:
            user_msg = f"{history_block}\n\nNew question: {question}"

        raw = call_llm(
            system=INTENT_SYSTEM,
            user=user_msg,
            max_tokens=10,
        ).strip().lower()

        if raw in ("structured", "document", "hybrid", "out_of_scope"):
            return raw  # type: ignore[return-value]

        # Fallback: guess from keywords
        doc_keywords = ("contract", "clause", "penalty", "terminat", "notice",
                        "payment term", "auto-renew", "force majeure", "sla definition")
        struct_keywords = ("spend", "po", "purchase order", "sla %", "compliance",
                           "renewal date", "price spike", "invoice", "top supplier",
                           "how many", "total", "average", "which region")
        ql = question.lower()
        has_doc = any(k in ql for k in doc_keywords)
        has_struct = any(k in ql for k in struct_keywords)
        if has_doc and has_struct:
            return "hybrid"
        if has_doc:
            return "document"
        if has_struct:
            return "structured"
        return "structured"   # default to structured for procurement questions

    # ── Coreference resolution ────────────────────────────────────────────────

    def _resolve_question(self, question: str) -> str:
        """Rewrite follow-up questions to be self-contained if history exists."""
        if not self.history:
            return question

        # Only rewrite if the question seems to reference prior context
        pronouns = ("it", "they", "them", "their", "that", "those", "this",
                    "same", "above", "previous", "mentioned")
        if not any(p in question.lower().split() for p in pronouns):
            return question

        history_block = _format_history(self.history)
        system = (
            "You are a query resolver. Given prior conversation and a follow-up question, "
            "rewrite the question to be fully self-contained (no pronouns or references "
            "to prior turns). Return ONLY the rewritten question, nothing else."
        )
        user_msg = f"{history_block}\n\nFollow-up: {question}"
        resolved = call_llm(system=system, user=user_msg, max_tokens=120).strip()
        logger.debug(f"Resolved question: {resolved}")
        return resolved

    # ── Router ────────────────────────────────────────────────────────────────

    def _route(self, question: str, intent: Intent) -> OrchestratorResponse:
        if intent == "out_of_scope":
            return OrchestratorResponse(
                question=question,
                answer=OUT_OF_SCOPE_MSG,
                intent=intent,
            )

        elif intent == "structured":
            result = self.structured_agent.run(question)
            return OrchestratorResponse(
                question=question,
                answer=result.answer,
                intent=intent,
                sql_used=result.sql_used,
                structured_rows=result.raw_rows,
                citations=result.citations,
                error=result.error,
            )

        elif intent == "document":
            result = self.document_agent.run(question)
            return OrchestratorResponse(
                question=question,
                answer=result.answer,
                intent=intent,
                contract_chunks=result.chunks_used,
                citations=result.citations,
                error=result.error,
            )

        elif intent == "hybrid":
            return self._run_hybrid(question)

        else:
            # Should never reach here
            return OrchestratorResponse(
                question=question,
                answer="Unable to classify this question. Please rephrase.",
                intent="out_of_scope",
            )

    # ── Hybrid path ───────────────────────────────────────────────────────────

    def _run_hybrid(self, question: str) -> OrchestratorResponse:
        """Run both agents concurrently and synthesise a combined answer."""
        logger.info("Hybrid path: running both agents in parallel…")

        # Run agents concurrently — each is an independent I/O-bound call
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_struct = pool.submit(self.structured_agent.run, question)
            future_doc    = pool.submit(self.document_agent.run, question)
            struct_result = future_struct.result()
            doc_result    = future_doc.result()

        logger.info("Hybrid path: both agents complete")

        # Build combined citations (deduplicated)
        all_citations: list[str] = []
        seen: set[str] = set()
        for c in struct_result.citations + doc_result.citations:
            if c not in seen:
                seen.add(c)
                all_citations.append(c)

        # If both agents failed
        if struct_result.error and doc_result.error:
            return OrchestratorResponse(
                question=question,
                answer="Both data sources returned errors. Please try again.",
                intent="hybrid",
                error=f"Structured: {struct_result.error} | Document: {doc_result.error}",
            )

        # Synthesise
        user_msg = (
            f"User question: {question}\n\n"
            f"STRUCTURED DATA ANSWER:\n{struct_result.answer}\n\n"
            f"CONTRACT DOCUMENT ANSWER:\n{doc_result.answer}"
        )
        combined_answer = call_llm(
            system=HYBRID_SYNTHESIS_SYSTEM,
            user=user_msg,
            max_tokens=700,
        )

        return OrchestratorResponse(
            question=question,
            answer=combined_answer,
            intent="hybrid",
            sql_used=struct_result.sql_used,
            structured_rows=struct_result.raw_rows,
            contract_chunks=doc_result.chunks_used,
            citations=all_citations,
        )


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    orchestrator = ProcurementOrchestrator()

    demo = [
        # Structured
        "Who are my top 5 suppliers by spend?",
        # Follow-up (tests coreference resolution)
        "What region are they in?",
        "What happens if a supplier misses SLA for 3 consecutive months?",
        # Hybrid
        "Which supplier has the worst SLA compliance and what does their contract say about penalties?",
        # Out of scope
        "What's the best recipe for pasta carbonara?",
    ]

    for q in demo:
        r = orchestrator.run(q)
        r.display()
