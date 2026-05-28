"""
synthetic/generate_data.py
Generates realistic procurement synthetic data:
  - 28 suppliers  → data/supplier_master/suppliers.csv
  - 500 POs       → data/purchase_orders/purchase_orders.csv
  - 28 contracts  → data/contracts/contract_SUP<NNNN>.txt

Run: python -m synthetic.generate_data
"""

import random
import sys
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
from faker import Faker

# Allow running as a module from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import SUPPLIERS_CSV, PO_CSV, CONTRACTS_DIR
from utils.logger import logger

fake = Faker()
random.seed(42)
Faker.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────
CATEGORIES = [
    "Raw Materials", "Packaging", "Logistics", "IT Services",
    "Facilities", "MRO", "Professional Services", "Chemicals",
]
REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
PAYMENT_TERMS = [30, 45, 60, 90]
TODAY = date.today()

# ── 1. Suppliers ──────────────────────────────────────────────────────────────
def generate_suppliers(n: int = 28) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        contract_end = TODAY + timedelta(days=random.randint(-30, 365))
        days_to_renewal = (contract_end - TODAY).days
        rows.append({
            "supplier_id": f"SUP{i:04d}",
            "supplier_name": fake.company(),
            "category": random.choice(CATEGORIES),
            "region": random.choice(REGIONS),
            "contract_start_date": (contract_end - timedelta(days=random.randint(365, 1095))).isoformat(),
            "contract_end_date": contract_end.isoformat(),
            "days_to_renewal": days_to_renewal,
            "annual_spend_usd": round(random.uniform(50_000, 5_000_000), 2),
            "payment_terms_days": random.choice(PAYMENT_TERMS),
            "sla_compliance_pct": round(random.uniform(72.0, 99.5), 1),
            "preferred_vendor": random.choice([True, False]),
            "contact_email": fake.company_email(),
        })
    df = pd.DataFrame(rows)
    SUPPLIERS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUPPLIERS_CSV, index=False)
    logger.success(f"✅ Saved {len(df)} suppliers → {SUPPLIERS_CSV}")
    return df


# ── 2. Purchase Orders ────────────────────────────────────────────────────────
ITEMS = [
    "Steel coils", "Cardboard boxes", "Freight forwarding",
    "Cloud hosting", "Cleaning supplies", "HVAC maintenance",
    "Legal consulting", "Solvent chemicals", "Packaging film",
    "Laptop computers", "Office furniture", "Safety equipment",
    "Lubricants", "Network switches", "Security services",
]

def generate_purchase_orders(suppliers_df: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    supplier_ids = suppliers_df["supplier_id"].tolist()
    rows = []
    for i in range(1, n + 1):
        order_date = TODAY - timedelta(days=random.randint(0, 730))
        sla_met = random.random() > 0.18           # ~18% miss SLA
        delivery_delay = 0 if sla_met else random.randint(1, 21)
        price_spike = random.random() < 0.08       # ~8% spike
        rows.append({
            "po_id": f"PO{i:05d}",
            "supplier_id": random.choice(supplier_ids),
            "item_description": random.choice(ITEMS),
            "quantity": random.randint(1, 1000),
            "unit_price_usd": round(random.uniform(5, 5000), 2),
            "total_amount_usd": round(random.uniform(500, 500_000), 2),
            "order_date": order_date.isoformat(),
            "expected_delivery_days": random.choice([7, 14, 21, 30]),
            "sla_met": sla_met,
            "delivery_delay_days": delivery_delay,
            "price_spike_flag": price_spike,
            "invoice_matched": random.random() > 0.05,
        })
    df = pd.DataFrame(rows)
    PO_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PO_CSV, index=False)
    logger.success(f"✅ Saved {len(df)} purchase orders → {PO_CSV}")
    return df


# ── 3. Contract documents ─────────────────────────────────────────────────────
CONTRACT_TEMPLATE = """\
MASTER SUPPLY AGREEMENT

Contract ID  : {contract_id}
Supplier     : {supplier_name} ({supplier_id})
Category     : {category}
Region       : {region}
Effective    : {contract_start_date}
Expiry       : {contract_end_date}
Payment Terms: Net {payment_terms_days} days

═══════════════════════════════════════════════════════════════

1. SCOPE OF SUPPLY
   {supplier_name} ("Supplier") agrees to supply {category} goods and services
   to Acme Corp ("Buyer") as described in individual Purchase Orders (POs)
   issued under this agreement.

2. PRICING & PAYMENT
   2.1 Unit prices are fixed for the duration of this agreement unless a
       written amendment is executed by both parties.
   2.2 Invoices are payable within {payment_terms_days} calendar days of receipt.
   2.3 Late payments accrue interest at 1.5% per month on outstanding balances.
   2.4 Price escalation exceeding 5% in any rolling 3-month period requires
       written justification and Buyer approval prior to invoicing.

3. DELIVERY & SLA OBLIGATIONS
   3.1 Supplier shall meet the delivery schedule agreed per PO.
   3.2 SLA compliance target: {sla_compliance_pct}% on-time delivery measured monthly.
   3.3 Delivery delays beyond 7 calendar days trigger a penalty of 2% of the
       affected PO value per week, capped at 10% of total PO value.
   3.4 Chronic non-performance (3 consecutive months below SLA target) gives
       Buyer the right to terminate with 30 days written notice.

4. QUALITY & INSPECTION
   4.1 Buyer may inspect goods at point of manufacture or delivery.
   4.2 Non-conforming goods must be replaced within 14 days at Supplier's cost.
   4.3 Supplier maintains ISO 9001 certification throughout the term.

5. INTELLECTUAL PROPERTY
   5.1 All custom tooling, designs, and specifications provided by Buyer remain
       Buyer's sole property.
   5.2 Supplier grants Buyer a non-exclusive licence to use Supplier's
       background IP solely for use of the supplied goods.

6. CONFIDENTIALITY
   Both parties agree to keep the terms of this agreement and any shared
   technical or commercial information strictly confidential for 5 years
   post-expiry.

7. FORCE MAJEURE
   Neither party is liable for delays caused by events beyond reasonable
   control (natural disasters, strikes, government actions). The affected
   party must notify the other within 48 hours.

8. TERMINATION
   8.1 Either party may terminate for material breach with 30 days written
       notice if the breach is not cured within the notice period.
   8.2 Buyer may terminate for convenience with 60 days written notice.
   8.3 Upon termination, all outstanding POs must be fulfilled or cancelled
       within 30 days.

9. GOVERNING LAW
   This agreement is governed by the laws of the State of Delaware, USA.
   Disputes shall be resolved by binding arbitration under AAA rules.

10. RENEWAL
    This agreement auto-renews for successive 12-month terms unless either
    party provides written notice of non-renewal at least 90 days before expiry.

═══════════════════════════════════════════════════════════════
Signed for and on behalf of:

Acme Corp                          {supplier_name}
_______________________            _______________________
Chief Procurement Officer          Authorised Signatory
Date: {contract_start_date}        Date: {contract_start_date}
"""

def generate_contracts(suppliers_df: pd.DataFrame) -> None:
    CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    for _, row in suppliers_df.iterrows():
        text = CONTRACT_TEMPLATE.format(
            contract_id=f"CTR-{row['supplier_id']}-2024",
            **row.to_dict(),
        )
        path = CONTRACTS_DIR / f"contract_{row['supplier_id']}.txt"
        path.write_text(text, encoding="utf-8")
    logger.success(f"✅ Saved {len(suppliers_df)} contract files → {CONTRACTS_DIR}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("🚀 Starting synthetic data generation…")
    suppliers = generate_suppliers(28)
    generate_purchase_orders(suppliers, 500)
    generate_contracts(suppliers)
    logger.info("✅ All synthetic data generated successfully.")
