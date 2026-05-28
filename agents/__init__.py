"""
agents/ — Procurement CoPilot agent package.

Public exports:
  ProcurementOrchestrator  — top-level router (use this in the UI)
  StructuredQueryAgent     — NL → SQL → SQLite
  DocumentRetrievalAgent   — semantic search → ChromaDB RAG
  OrchestratorResponse     — unified response dataclass

Requires: chromadb, sentence-transformers, anthropic (or boto3 for Bedrock).
Install with:  pip install -r requirements.txt
"""

try:
    from agents.orchestrator import ProcurementOrchestrator, OrchestratorResponse
    from agents.structured_query_agent import StructuredQueryAgent
    from agents.document_retrieval_agent import DocumentRetrievalAgent

    __all__ = [
        "ProcurementOrchestrator",
        "OrchestratorResponse",
        "StructuredQueryAgent",
        "DocumentRetrievalAgent",
    ]
except ImportError as _e:  # pragma: no cover
    import warnings
    warnings.warn(
        f"Procurement agents could not be fully loaded ({_e}). "
        "Run: pip install -r requirements.txt",
        ImportWarning,
        stacklevel=2,
    )
