Procurement Copilot — Build Log #2 🔨

I asked my AI: "What's the early payment discount?"
It answered — confidently — about expedited shipping.

That's when I understood: RAG defaults are built for generic text.
Procurement contracts are not generic text.

Two decisions I had to make. Here's what they were — and why. 👇

─────────────────────────
📌 Decision 1: How to chunk the document

The default: split every document into 512-token blocks.

The problem: a token counter has no concept of document structure.
Section 2.1 — "Early Payment Terms" — sits inside Section 2 — "Pricing."
Fixed-size chunking doesn't know that.
It slices across boundaries. Merges unrelated clauses into one block.
The chunk carries words. Not hierarchy. Not location.

In a supplier contract, structure is the signal.
A penalty clause and a discount clause can appear 20 lines apart.
If they land in the same block, the retriever can't distinguish them.

The fix: split by document structure, not token count.
→ Section headers define the boundaries
→ Paragraphs split within each section
→ Sentences only when a paragraph is still too large

Every chunk now carries its full path:
"2. Pricing → 2.1 Early Payment Terms"

The retriever knows where a chunk lives — not just what it says.

💡 For hierarchical documents, structure is the right unit of chunking.
Token count is a proxy that destroys the signal you actually need.

─────────────────────────
📌 Decision 2: What to embed

Fixing structure solved location. It didn't solve meaning.

When a chunk is embedded, the model sees only that chunk.
No surrounding document. No contract type. No business context.

"2% discount if paid within 10 days."
Embedded in isolation — that's all it is.
The vector captures those 15 words. Nothing more.

Query: "What are our early settlement terms?"
Retriever returns nothing.
Not because the answer isn't there.
Because the vector was built without knowing what it was part of.

The root cause isn't vocabulary mismatch. It's context-free indexing.

The fix: before embedding, run a separate LLM pass over each chunk.
It reads the full document and writes 2-3 sentences of context:
→ Where the chunk sits in the document hierarchy
→ What obligation or concept it captures
→ How it connects to the document's overall purpose

That context is prepended to the chunk before embedding.

Same 15 words. Completely different vector.
The model now stores: "This is an early payment discount clause in the Payment Terms section of a Master Supply Agreement governing supplier commercial terms."

Retrieval on ambiguous queries improved immediately.

💡 Enrichment is not preprocessing.
It's the step that teaches the vector what the text means — before it's stored.

─────────────────────────

Two decisions. Both invisible in standard RAG tutorials.
Both critical for enterprise documents.

Next: how I search across these chunks.
One query is never enough — and the fix is counterintuitive.

What's your experience building RAG on structured documents? 👇

#SupplyChain #Procurement #RAG #AgenticAI #GenerativeAI #ProcurementTech #BuildingInPublic
