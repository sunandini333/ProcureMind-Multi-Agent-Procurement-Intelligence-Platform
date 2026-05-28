"""
utils/config.py — Central configuration for Procurement Copilot.
All paths are relative to the project root so the app is portable.

Secret resolution order (highest priority first):
  1. Streamlit secrets  (st.secrets)  — used on Streamlit Community Cloud
  2. Environment variables / .env     — used for local development
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _secret(key: str, default: str = "") -> str:
    """Read a secret from Streamlit secrets first, then env vars."""
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


# ── Project root (one level above this file) ─────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = _secret("LLM_PROVIDER", "anthropic")            # "anthropic" | "bedrock"
ANTHROPIC_API_KEY: str = _secret("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

# AWS Bedrock (swap in for Dell environment)
AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID: str = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

# ── Vector store ─────────────────────────────────────────────────────────────
VECTOR_STORE_TYPE: str = "chroma"                                    # only chroma for now
CHROMA_PERSIST_DIR: Path = ROOT / "knowledge_base" / "vector_store"
CHROMA_COLLECTION_NAME: str = "procurement_contracts"

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CHUNK_SIZE: int = 1000         # characters — large enough to hold a full contract clause
CHUNK_OVERLAP: int = 150       # characters — enough to bridge clause boundaries
TOP_K: int = 5
MIN_RELEVANCE_PCT: float = 50.0  # discard chunks below this cosine-similarity threshold

# ── Structured DB ────────────────────────────────────────────────────────────
SQLITE_PATH: Path = ROOT / "knowledge_base" / "structured_db" / "procurement.db"
SQLITE_URL: str = f"sqlite:///{SQLITE_PATH}"

# ── Data paths ───────────────────────────────────────────────────────────────
SUPPLIERS_CSV: Path = ROOT / "data" / "supplier_master" / "suppliers.csv"
PO_CSV: Path = ROOT / "data" / "purchase_orders" / "purchase_orders.csv"
CONTRACTS_DIR: Path = ROOT / "data" / "contracts"

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = _secret("LOG_LEVEL", "INFO")
LOG_DIR: Path = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
