"""
agents/structured_query_agent.py
Natural language → SQL → executed results → synthesised answer.

Flow:
  1. User asks a question in plain English.
  2. Claude generates the correct SQL given the schema context.
  3. SQL is validated (read-only guard) and executed against SQLite.
  4. Claude synthesises a clean prose answer from the raw rows.
  5. Returns AgentResponse with answer, SQL used, raw rows, and citations.

Run directly for a quick demo:
  python -m agents.structured_query_agent
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.llm_client import call_llm
from utils.db_helper import get_db_connection
from utils.logger import logger


# ── Schema context fed to Claude ──────────────────────────────────────────────

SCHEMA_CONTEXT = """
You have access to a SQLite procurement database with the following schema:

TABLE: suppliers
  supplier_id         TEXT  PRIMARY KEY  (e.g. "SUP0001")
  supplier_name       TEXT
  category            TEXT  (Raw Materials | Packaging | Logistics | IT Services |
                             Facilities | MRO | Professional Services | Chemicals)
  region              TEXT  (North America | EMEA | APAC | LATAM)
  contract_start_date TEXT  (ISO date)
  contract_end_date   TEXT  (ISO date)
  days_to_renewal     INT   (negative = already expired)
  annual_spend_usd    REAL
  payment_terms_days  INT   (30 | 45 | 60 | 90)
  sla_compliance_pct  REAL  (0-100)
  preferred_vendor    BOOL
  contact_email       TEXT

TABLE: purchase_orders
  po_id               TEXT  PRIMARY KEY  (e.g. "PO00001")
  supplier_id         TEXT  (FK → suppliers.supplier_id)
  item_description    TEXT
  quantity            INT
  unit_price_usd      REAL
  total_amount_usd    REAL
  order_date          TEXT  (ISO date)
  expected_delivery_days INT
  sla_met             BOOL  (1=on time, 0=late)
  delivery_delay_days INT   (0 if on time)
  price_spike_flag    BOOL  (1=unusual price increase detected)
  invoice_matched     BOOL

VIEW: supplier_spend_summary   (pre-joined suppliers + purchase_orders aggregate)
  supplier_id, supplier_name, category, region,
  annual_spend_usd, sla_compliance_pct, days_to_renewal,
  total_po_count, total_po_spend, avg_po_value,
  sla_violations  (count of POs where sla_met=0),
  price_spikes    (count of POs where price_spike_flag=1)

VIEW: renewal_risk_suppliers   (suppliers with days_to_renewal BETWEEN -30 AND 90)
  supplier_id, supplier_name, category, region,
  contract_end_date, days_to_renewal, annual_spend_usd, sla_compliance_pct

IMPORTANT NOTES:
- Use supplier_spend_summary for spend/SLA aggregates — it's faster than joining manually.
- Use renewal_risk_suppliers for contract renewal questions.
- sla_violations in supplier_spend_summary counts POs where sla_met=0.
- days_to_renewal < 0 means the contract has already expired.
- Monetary values are in USD.
- Dates are stored as TEXT in ISO 8601 format (YYYY-MM-DD); use strftime or string comparison.
"""

SQL_GENERATION_SYSTEM = f"""You are a precise SQL expert for a procurement analytics system.

{SCHEMA_CONTEXT}

RULES:
1. Return ONLY the SQL query — no markdown, no code fences, no explanation.
2. Use only SELECT statements (never INSERT, UPDATE, DELETE, DROP, CREATE).
3. Always use LIMIT (max 50) unless the question explicitly asks for all records.
4. Use ROUND(value, 2) for monetary amounts.
5. Alias columns clearly (e.g. total_po_spend AS spend_usd).
6. If the question cannot be answered from this schema, return exactly: CANNOT_ANSWER
"""

SYNTHESIS_SYSTEM = """You are a procurement analyst assistant. Your job is to turn raw SQL
query results into clear, concise, business-friendly answers.

Guidelines:
- Lead with the direct answer to the question (1-2 sentences).
- Then elaborate with key numbers and insights from the data.
- Highlight any risks, anomalies, or action items if present.
- Keep the tone professional but conversational.
- Do NOT mention SQL, databases, or technical implementation.
- Format monetary values with $ and commas (e.g. $1,234,567).
- Format percentages with one decimal place (e.g. 94.3%).
- If the result set is empty, say so clearly and suggest why.
"""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    question: str
    answer: str
    sql_used: str
    raw_rows: list[dict[str, Any]]
    citations: list[str] = field(default_factory=list)
    error: str | None = None

    def display(self) -> None:
        """Pretty-print to terminal."""
        print(f"\n{'─'*65}")
        print(f"❓ {self.question}")
        print(f"{'─'*65}")
        if self.error:
            print(f"⚠️  Error: {self.error}")
        else:
            print(f"\n💬 {self.answer}")
            print(f"\n📊 SQL used:\n   {self.sql_used}")
            print(f"\n📋 Raw rows ({len(self.raw_rows)}):")
            for r in self.raw_rows[:10]:
                print(f"   {r}")
            if len(self.raw_rows) > 10:
                print(f"   … and {len(self.raw_rows)-10} more")
            if self.citations:
                print(f"\n📎 Citations: {', '.join(self.citations)}")
        print()


# ── Agent class ───────────────────────────────────────────────────────────────

class StructuredQueryAgent:
    """
    Converts natural language procurement questions into SQL,
    executes them, and synthesises a clean answer using Claude.
    """

    # Only allow read-only SQL
    _FORBIDDEN = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|ATTACH)\b",
        re.IGNORECASE,
    )

    def run(self, question: str) -> AgentResponse:
        """
        Full pipeline: question → SQL → execute → synthesise → AgentResponse.
        """
        logger.info(f"StructuredQueryAgent: '{question[:80]}'")

        # Step 1 — Generate SQL
        sql = self._generate_sql(question)
        if sql == "CANNOT_ANSWER":
            return AgentResponse(
                question=question,
                answer="This question cannot be answered from the structured procurement database. "
                       "It may require document search — try asking about contract terms instead.",
                sql_used="CANNOT_ANSWER",
                raw_rows=[],
            )

        # Step 2 — Safety check
        if self._FORBIDDEN.search(sql):
            logger.warning(f"Blocked unsafe SQL: {sql[:100]}")
            return AgentResponse(
                question=question,
                answer="Query blocked: only read-only SELECT statements are permitted.",
                sql_used=sql,
                raw_rows=[],
                error="Unsafe SQL blocked",
            )

        # Step 3 — Execute
        rows, exec_error = self._execute_sql(sql)
        if exec_error:
            # Give Claude one retry with the error message
            logger.warning(f"SQL error on first attempt: {exec_error} — retrying…")
            sql = self._generate_sql(question, error_hint=exec_error)
            rows, exec_error = self._execute_sql(sql)
            if exec_error:
                return AgentResponse(
                    question=question,
                    answer=f"Could not execute the query after retry. Error: {exec_error}",
                    sql_used=sql,
                    raw_rows=[],
                    error=exec_error,
                )

        # Step 4 — Synthesise answer
        answer = self._synthesise(question, sql, rows)

        # Step 5 — Build citations (supplier IDs / PO IDs found in results)
        citations = self._extract_citations(rows)

        return AgentResponse(
            question=question,
            answer=answer,
            sql_used=sql,
            raw_rows=rows,
            citations=citations,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_sql(self, question: str, error_hint: str | None = None) -> str:
        user_msg = f"Question: {question}"
        if error_hint:
            user_msg += f"\n\nPrevious attempt failed with: {error_hint}\nPlease fix the SQL."

        sql = call_llm(system=SQL_GENERATION_SYSTEM, user=user_msg, max_tokens=512)

        # Strip any accidental markdown fences
        sql = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE).strip().rstrip(";")
        logger.debug(f"Generated SQL: {sql}")
        return sql

    def _execute_sql(self, sql: str) -> tuple[list[dict], str | None]:
        try:
            conn = get_db_connection()
            rows = conn.execute(sql).fetchall()
            conn.close()
            return [dict(r) for r in rows], None
        except Exception as e:
            return [], str(e)

    def _synthesise(self, question: str, sql: str, rows: list[dict]) -> str:
        rows_text = "\n".join(str(r) for r in rows[:30])
        if not rows:
            rows_text = "(no rows returned)"

        user_msg = (
            f"Question: {question}\n\n"
            f"SQL used:\n{sql}\n\n"
            f"Results ({len(rows)} rows):\n{rows_text}"
        )
        return call_llm(system=SYNTHESIS_SYSTEM, user=user_msg, max_tokens=600)

    def _extract_citations(self, rows: list[dict]) -> list[str]:
        citations = []
        for row in rows:
            if sid := row.get("supplier_id"):
                cite = f"suppliers/{sid}"
                if cite not in citations:
                    citations.append(cite)
            if po := row.get("po_id"):
                cite = f"PO/{po}"
                if cite not in citations:
                    citations.append(cite)
        return citations[:10]  # cap at 10


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = StructuredQueryAgent()

    demo_questions = [
        "Who are my top 5 suppliers by spend?",
        "Which suppliers are renewal risks this quarter?",
        "Which suppliers have more than 3 SLA violations?",
        "Show me all POs with price spikes and their suppliers",
        "What is the average SLA compliance by region?",
    ]

    for q in demo_questions:
        result = agent.run(q)
        result.display()
