Procurement Copilot — Build Log #3 🔨

The chunks were right. The retriever still missed them.

I searched for "payment discount terms."
The answer was in a clause titled "Early Settlement Incentive."
Same concept. Different words.
Exact vector match: zero.

That's the retrieval problem. Here's how I solved it — twice. 👇

─────────────────────────
📌 Decision 3: How to search

The default: embed the query, find the nearest vectors, return top results.

The problem: a single query is a single perspective.
The embedding encodes exactly the words you typed.
If the contract uses different terminology, the vector distance is large — even when the meaning is identical.

Procurement documents are written by lawyers, not search engineers.
"Early payment discount" and "early settlement incentive" and "2/10 net 30" all mean the same thing.
A single query embedding catches one phrasing. It misses the rest.

The fix: don't send one query. Send many.
→ An LLM generates 3 query variants from the original question
→ Each variant retrieves its own result set
→ All result sets are merged using Reciprocal Rank Fusion (RRF)

RRF scores each chunk based on its rank across all result sets.
A chunk that appears in the top results for multiple query variants scores significantly higher than one that matches only one phrasing.

The retriever now searches from multiple angles simultaneously.
Terminology variation stops being a retrieval failure.

💡 One query = one perspective. For domain documents with inconsistent language, multi-query retrieval is not an optimisation — it's a requirement.

─────────────────────────
📌 Decision 4: How to rank

The default: return the top results from retrieval, ranked by cosine similarity.

The problem: cosine similarity measures vector distance, not relevance.
The bi-encoder embeds the query and each chunk independently — then compares the two vectors.
The model never sees them together.
A chunk about "payment obligations" and a chunk about "early settlement discount" can have nearly identical vectors.
The score doesn't tell you which one actually answers the question.

So after retrieval, I had 12 candidates.
Several were genuinely useful. Several were topically adjacent but not the answer.
The similarity scores didn't separate them.

The fix: a second model reads each (query, chunk) pair jointly.
Both inputs in the same forward pass.
It doesn't measure distance. It scores relevance directly.
→ Retrieve wide: top 12 from RAG-Fusion
→ Rerank tight: cross-encoder scores all 12 against the original query
→ Return top 4 — the ones that actually answer the question

The pipeline defaults to cross-encoder on.
Not because it's fast — it's slower than the bi-encoder.
Because in procurement, one wrong clause changes a decision.
Speed is a trade-off. Precision is not.

💡 Retrieval and ranking are different jobs. The model that finds candidates cannot reliably rank them. Use two models — one for each job.

─────────────────────────

Four decisions across two posts.
Each one addresses a specific failure mode — structural, semantic, linguistic, precision.

Next: the full pipeline running on a real supplier contract.
Real procurement questions. Actual answers. Side-by-side comparison against the baseline.

That's where it gets concrete. 👇

#SupplyChain #Procurement #RAG #AgenticAI #GenerativeAI #ProcurementTech #BuildingInPublic
