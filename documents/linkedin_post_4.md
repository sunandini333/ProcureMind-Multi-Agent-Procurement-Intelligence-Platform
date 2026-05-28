Procurement Copilot — Build Log #4 🔨

Post 2 opened with a failure.
I asked: "What's the early payment discount?"
The AI answered — confidently — about expedited shipping.

This post closes that loop.

─────────────────────────
📌 The test

Four decisions. Two posts.
Hierarchical chunking. Contextual enrichment. RAG-Fusion. Cross-encoder reranking.

One question: does it actually work?

I ran three procurement queries against two configurations.
Baseline: fixed-size chunks, single vector query, no reranking.
Full stack: all four decisions active.
Same LLM generating the answer in both cases.

─────────────────────────
📌 The results

Q1: "What is the early payment discount?"

Baseline → Returned a passage about expedited delivery surcharges.
Confident. Wrong. The fixed chunk merged payment and delivery clauses together.

Full stack → "Buyer receives a 2% discount if payment is settled within 10 days of invoice date (2/10 Net 30)."
Source: 3. Payment Terms → 3.2 Early Payment Discount

─────────────────────────

Q2: "What happens if a shipment arrives without a Certificate of Conformance?"

Baseline → Retrieved the quality standards clause. Mentioned ISO 9001 requirements.
Technically related. Not the answer.

Full stack → "Shipments received without a CoC may be placed in quarantine at Buyer's discretion pending documentation."
Source: 5. Quality and Compliance → 5.3 Certificate of Conformance

─────────────────────────

Q3: "What are the volume discount tiers and how do I qualify?"

Baseline → Returned base unit pricing. The 10,000+ tier qualification requirement was missing.

Full stack → Both tiers returned with conditions:
→ 5,000–9,999 units: 5% applied automatically
→ 10,000+ units: 10%, requires a 30-day volume forecast prior to order
Source: 2. Pricing → 2.2 Volume Discounts

─────────────────────────
📌 The takeaway

Three questions. Three retrieval failures in the baseline.
Three precise answers in the full stack.

The LLM was identical in both runs.
The difference was entirely in what it was given to work with.

Garbage in, garbage out is too simple.
The real version: structure in, precision out.

─────────────────────────
📌 What the pipeline can't do

It answers questions. One at a time. In isolation.

Ask the same thing twice — it retrieves from scratch.
Reference a previous answer — it has no memory of it.
Span three supplier contracts — it doesn't know they're related.
Flag a risk before you ask — it waits.

This is a retrieval system. Not an agent.

The next layer is different:
→ Memory across conversation turns
→ Reasoning across multiple documents
→ Actions — not just answers

That's Build Log #5.

#SupplyChain #Procurement #RAG #AgenticAI #GenerativeAI #ProcurementTech #BuildingInPublic
