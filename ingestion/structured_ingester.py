"""
ingestion/structured_ingester.py
Loads suppliers.csv + purchase_orders.csv into SQLite and creates analytical views.

Run: python -m ingestion.structured_ingester
"""

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import SQLITE_URL, SQLITE_PATH, SUPPLIERS_CSV, PO_CSV
from utils.logger import logger


def load_tables(engine) -> None:
    suppliers = pd.read_csv(SUPPLIERS_CSV)
    pos = pd.read_csv(PO_CSV)

    suppliers.to_sql("suppliers", engine, if_exists="replace", index=False)
    logger.success(f"✅ Loaded {len(suppliers)} rows → suppliers table")

    pos.to_sql("purchase_orders", engine, if_exists="replace", index=False)
    logger.success(f"✅ Loaded {len(pos)} rows → purchase_orders table")


def create_views(engine) -> None:
    views = {
        "supplier_spend_summary": """
            CREATE VIEW IF NOT EXISTS supplier_spend_summary AS
            SELECT
                s.supplier_id,
                s.supplier_name,
                s.category,
                s.region,
                s.annual_spend_usd,
                s.sla_compliance_pct,
                s.days_to_renewal,
                COUNT(po.po_id)                             AS total_po_count,
                ROUND(SUM(po.total_amount_usd), 2)          AS total_po_spend,
                ROUND(AVG(po.total_amount_usd), 2)          AS avg_po_value,
                SUM(CASE WHEN po.sla_met = 0 THEN 1 ELSE 0 END)  AS sla_violations,
                SUM(CASE WHEN po.price_spike_flag = 1 THEN 1 ELSE 0 END) AS price_spikes
            FROM suppliers s
            LEFT JOIN purchase_orders po ON s.supplier_id = po.supplier_id
            GROUP BY s.supplier_id
        """,
        "renewal_risk_suppliers": """
            CREATE VIEW IF NOT EXISTS renewal_risk_suppliers AS
            SELECT
                supplier_id,
                supplier_name,
                category,
                region,
                contract_end_date,
                days_to_renewal,
                annual_spend_usd,
                sla_compliance_pct
            FROM suppliers
            WHERE days_to_renewal BETWEEN -30 AND 90
            ORDER BY days_to_renewal ASC
        """,
    }

    with engine.connect() as conn:
        for name, ddl in views.items():
            conn.execute(text(f"DROP VIEW IF EXISTS {name}"))
            conn.execute(text(ddl))
            conn.commit()
            logger.success(f"✅ View created: {name}")


def validate(engine) -> None:
    queries = {
        "Total suppliers": "SELECT COUNT(*) FROM suppliers",
        "Total POs": "SELECT COUNT(*) FROM purchase_orders",
        "Renewal risks (next 90 days)": "SELECT COUNT(*) FROM renewal_risk_suppliers",
        "Top supplier by spend": (
            "SELECT supplier_name, total_po_spend FROM supplier_spend_summary "
            "ORDER BY total_po_spend DESC LIMIT 1"
        ),
    }
    with engine.connect() as conn:
        for label, sql in queries.items():
            result = conn.execute(text(sql)).fetchone()
            logger.info(f"  {label}: {result}")


if __name__ == "__main__":
    import shutil
    import tempfile
    # SQLite needs reliable file-locking; some sync'd filesystems (OneDrive, virtiofs)
    # don't play nicely with it. Build in the OS temp dir, then copy into the workspace.
    TMP_DIR = Path(tempfile.gettempdir())
    LOCAL_DB = TMP_DIR / "procurement.db"
    if LOCAL_DB.exists():
        LOCAL_DB.unlink()

    logger.info(f"🗄️  Starting structured ingestion (building in {TMP_DIR})…")
    tmp_url = f"sqlite:///{LOCAL_DB}"
    engine = create_engine(tmp_url)
    load_tables(engine)
    create_views(engine)
    validate(engine)

    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SQLITE_PATH.exists():
        SQLITE_PATH.unlink()
    # Also clean up any stale journal file from a previous interrupted run
    stale_journal = SQLITE_PATH.with_suffix(SQLITE_PATH.suffix + "-journal")
    if stale_journal.exists():
        stale_journal.unlink()
    shutil.copy2(LOCAL_DB, SQLITE_PATH)
    logger.success(f"✅ DB copied to workspace: {SQLITE_PATH} ({SQLITE_PATH.stat().st_size:,} bytes)")
    logger.info("✅ Structured ingestion complete.")
