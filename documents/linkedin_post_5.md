Procurement Copilot — Build Log #5 🔨

Build Log #4 ended with a confession.

"It answers questions. One at a time. In isolation."
"This is a retrieval system. Not an agent."

This post is where that changes. 👇

─────────────────────────
📌 Decision 5: How to route

The retrieval pipeline from logs 2–4 does one thing well: find relevant contract text.
But a procurement question isn't always a contract question.

"What is the penalty clause for late delivery?" → contract text.
"Which supplier has the worst on-time rate this quarter?" → spend database.
"Which supplier has the worst on-time rate — and what does our contract say we can do about it?" → both.

One retrieval system can't answer all three correctly.
Sending every question to the vector store means you'll hallucinate the SQL answers.
Sending every question to the database means you'll miss the clause that matters most.

The fix: a routing agent that reads the question and decides where it goes.

Before any retrieval, a fast Claude call classifies intent into one of four categories:
→ structured — needs the database (spend, SLA %, PO counts, renewal dates)
→ document  — needs contract text (clauses, penalties, termination, payment terms)
→ hybrid    — needs both
→ out_of_scope — politely declined

One word back. No extra tokens. The right door opens.

💡 Intent classification doesn't need to be a big model doing heavy reasoning. A small, focused prompt with exactly four possible outputs is fast, cheap, and reliable. Constrain the output space — you constrain the failure space.

─────────────────────────
📌 Decision 6: How to answer structured questions

The database has the numbers. The user asks in plain English.
The gap between them is SQL — and not everyone writes SQL.

The Structured Query Agent closes that gap:
1. The user's question arrives in natural language.
2. Claude sees the full schema (tables, columns, types, sample values) as context.
3. Claude writes the SQL. Validated as read-only before it runs.
4. SQLite executes it. Raw rows come back.
5. Claude converts rows into a plain-English answer with the relevant figures.

The user never sees SQL. They see: "Your top 5 suppliers by spend are…"

One failure mode I hit: the model sometimes writes queries that work but answer the wrong thing — especially with filters. ("Top suppliers by spend in APAC" would return global totals if the WHERE clause was missing.)

The fix was obvious in hindsight: include a handful of representative rows in the schema context, not just column names. The model needs to know what the data actually looks like to filter it correctly.

💡 NL-to-SQL is not about writing clever SQL. It's about giving the model enough context to write boring, correct SQL every time.

─────────────────────────
📌 Decision 7: How to handle follow-ups

A real conversation isn't a series of isolated questions.

"Which supplier has the worst SLA compliance?"
→ "What region are they in?"
→ "What does our contract say about penalties?"

The second and third questions are meaningless without the first answer as context.

Every agent in this system now receives the last 6 question-answer pairs before generating anything. But injecting raw history isn't enough — "What region are they in?" still needs to know that "they" means the supplier from the previous answer.

So before routing, a coreference resolution step rewrites ambiguous questions into self-contained ones:
"What region are they in?" → "What region is [Supplier X] in?"

The rewrite only runs if the question contains pronouns or reference words (it, they, that, same, above). No unnecessary LLM calls on clean questions.

💡 Conversation memory isn't just about storing history. It's about resolving what the user meant before you try to answer what they said.

─────────────────────────
📌 Decision 8: How to handle hybrid questions

The hardest questions span both worlds:
"Which supplier has the worst SLA compliance and what does their contract say about penalties?"

This requires the structured agent AND the document agent — in the right order, on the same question.

The implementation runs them in parallel using a thread pool (both are I/O-bound LLM calls).
Both results come back. A third Claude call synthesises a single executive-ready answer that connects the data insight to the relevant contract clause.

The result isn't two answers stitched together.
It's one answer a procurement manager can act on:
"Supplier X has the lowest SLA compliance at 71%. Section 8.2 of their contract allows you to issue a formal cure notice after two consecutive months of non-compliance, with termination rights after 90 days if unresolved."

That's not retrieval. That's reasoning.

─────────────────────────
📌 What it looks like now

All of this runs behind a Streamlit chat interface.

Each answer shows an intent badge — STRUCTURED / DOCUMENT / HYBRID — so you always know where the answer came from. SQL queries are expandable. Contract sources are expandable. Citations are linked to the relevant supplier and clause.

Ask a follow-up. It remembers. Change the supplier. It switches. Ask something irrelevant. It declines cleanly.

─────────────────────────

Four logs. Eight decisions. One working agent.

What's next: moving from a hand-rolled orchestrator to a proper agentic graph — state, branching, tool calls, and agents that can act, not just answer.

Build Log #6. 👇

#SupplyChain #Procurement #AgenticAI #RAG #GenerativeAI #ProcurementTech #BuildingInPublic #LLM
