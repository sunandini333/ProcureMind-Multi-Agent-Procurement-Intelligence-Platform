"""
demo.py
=======
Demonstrates the full RAG pipeline on a sample procurement document.

Run:
    cd rag/
    python demo.py

Make sure ANTHROPIC_API_KEY is set in your environment.
"""

from pipeline import RAGConfig, RAGPipeline

# ── Sample procurement document ────────────────────────────────────────────────
# Replace this with your own document text, or load from a file:
#   with open("my_contract.txt") as f:
#       DOCUMENT = f.read()

DOCUMENT = """
# Master Supply Agreement — Acme Corp & SupplierX Ltd

## 1. Parties and Purpose
This Master Supply Agreement ("Agreement") is entered into between Acme Corp
("Buyer") and SupplierX Ltd ("Seller"), effective January 1, 2025.

The Agreement governs the purchase of raw materials, components, and
sub-assemblies for Buyer's manufacturing operations in North America.

## 2. Pricing

### 2.1 Base Unit Pricing
Standard unit price is USD 45.00 per unit for orders up to 4,999 units.

### 2.2 Volume Discounts
- Orders 5,000–9,999 units: 5% discount applied automatically.
- Orders 10,000+ units: 10% discount. Buyer must submit a volume forecast
  30 days prior to order placement to qualify.

### 2.3 Price Review
Pricing is fixed for 12 months from the effective date. Either party may
request a price review with 60 days written notice after the initial term.
Adjustments are capped at ±8% per review cycle.

## 3. Payment Terms

### 3.1 Standard Terms
Payment is due Net-30 from the date of invoice. Invoices must reference the
Purchase Order number and include a Certificate of Conformance.

### 3.2 Early Payment Discount
Buyer receives a 2% discount if payment is settled within 10 days of invoice
date (2/10 Net 30).

### 3.3 Late Payment Penalty
Overdue amounts accrue interest at 1.5% per month (18% per annum).
Seller reserves the right to place orders on credit hold after 45 days past due.

## 4. Lead Times and Delivery

### 4.1 Standard Lead Time
Standard production and delivery lead time is 14 business days from Purchase
Order acknowledgement.

### 4.2 Expedited Orders
Expedited delivery (7 business days) is available at a 15% surcharge on the
unit price. Expedited requests must be confirmed in writing by 10:00 AM EST.

### 4.3 Incoterms
All shipments are governed by FCA (Seller's Warehouse, Detroit MI) per
Incoterms 2020. Title and risk pass to Buyer upon handover to the carrier.

### 4.4 Partial Shipments
Partial shipments are permitted with prior written approval. Each partial
shipment generates a separate invoice.

## 5. Quality and Compliance

### 5.1 Quality Standards
All goods must comply with ISO 9001:2015 and meet Buyer's published
specification sheet (Appendix A).

### 5.2 Acceptance and Rejection
Buyer has 10 business days post-receipt to inspect and accept or reject goods.
Rejection is valid only if the defect rate exceeds 0.5% of the shipment quantity
or a critical specification is breached.

### 5.3 Certificate of Conformance
Seller shall include a Certificate of Conformance (CoC) with each shipment.
Shipments received without a CoC may be placed in quarantine at Buyer's
discretion pending documentation.

## 6. Sustainability and Compliance

### 6.1 Conflict Minerals
Seller certifies compliance with Section 1502 of the Dodd-Frank Act regarding
conflict minerals. An annual CMRT report must be submitted by March 31 each year.

### 6.2 Environmental Standards
Seller must maintain ISO 14001 certification. Evidence of certification renewal
must be provided within 30 days of expiry.

## 7. Force Majeure
Neither party is liable for failure to perform obligations caused by events
beyond reasonable control, including natural disasters, pandemics, labor strikes,
government actions, port closures, or supply chain disruptions exceeding 30 days.

The affected party must notify the other in writing within 5 business days of the
triggering event. If the force majeure event persists beyond 90 days, either
party may terminate the Agreement with 30 days written notice.

## 8. Termination
Either party may terminate the Agreement for convenience with 90 days written
notice. Termination for cause (material breach uncured within 30 days) is
immediate upon written notice.

## 9. Governing Law
This Agreement is governed by the laws of the State of Michigan, USA.
Disputes are subject to binding arbitration under AAA Commercial Arbitration Rules.
"""

DOC_ID = "master_supply_agreement_acme_2025"


def main():
    # ── Configure ──────────────────────────────────────────────────────────────
    config = RAGConfig(
        chunker="hierarchical",           # Structure-aware chunking
        use_contextual_retrieval=True,    # Anthropic Contextual Retrieval
        retriever="fusion",               # RAG-Fusion with query expansion
        reranker="cross_encoder",         # Cross-encoder reranking
        collection_name="demo_procurement",
        persist_directory="./demo_chroma_db",
        n_retrieve=12,
        n_final=4,
    )

    pipeline = RAGPipeline(config)

    # ── Index ──────────────────────────────────────────────────────────────────
    pipeline.index_document(DOC_ID, DOCUMENT)

    # ── Query ──────────────────────────────────────────────────────────────────
    questions = [
        "What is the early payment discount and how do I qualify for it?",
        "What happens if a shipment arrives without a Certificate of Conformance?",
        "What are the volume discount tiers and qualification requirements?",
        "What are the force majeure notification and termination conditions?",
    ]

    for question in questions:
        result = pipeline.query(question)
        print(f"\n{'─'*60}")
        print(f"Q: {question}")
        print(f"\nA: {result['answer']}")
        print(f"\nSources used:")
        for src in result["sources"]:
            print(f"  • {src['breadcrumb']}  (score: {src['score']})")
        print()

    # ── Strategy comparison ────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STRATEGY COMPARISON — same question, different configs")
    pipeline.compare_strategies(
        "What are the payment terms?",
        strategies=[
            {"label": "Baseline  (fixed chunks, no rerank)",
             "chunker": "fixed_size", "retriever": "vector", "reranker": "none"},
            {"label": "Hierarchical + CrossEncoder",
             "chunker": "hierarchical", "retriever": "vector", "reranker": "cross_encoder"},
            {"label": "Full stack (hierarchical + fusion + cross-encoder)",
             "chunker": "hierarchical", "retriever": "fusion", "reranker": "cross_encoder"},
        ],
    )


if __name__ == "__main__":
    main()
