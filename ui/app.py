"""
ui/app.py — Procurement CoPilot · Streamlit Chat Interface  (Day 4)

Run from the project root:
    streamlit run ui/app.py

Features
────────
• Multi-turn chat with conversation memory (coreference resolution)
• Intent badge: STRUCTURED · DOCUMENT · HYBRID · OUT OF SCOPE
• SQL expander for structured answers
• Contract source expander for document answers
• Citations panel
• Sidebar: quick queries, session stats, model/provider info, reset
• Works with both Anthropic API and AWS Bedrock (toggle in .env)
"""

import sys
import time
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Page config (must be FIRST Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Procurement CoPilot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

from agents.orchestrator import ProcurementOrchestrator, OrchestratorResponse
from utils.config import LLM_PROVIDER, ANTHROPIC_MODEL, BEDROCK_MODEL_ID, ANTHROPIC_API_KEY


# ── Cold-start guard ──────────────────────────────────────────────────────────
# Shows a clear error on Streamlit Cloud if the API key secret is missing,
# rather than an obscure crash deep inside the agent stack.

def _check_secrets() -> bool:
    """Return True if the app is configured correctly, False otherwise."""
    if not ANTHROPIC_API_KEY and LLM_PROVIDER == "anthropic":
        st.error(
            "**🔑 ANTHROPIC_API_KEY is not set.**\n\n"
            "On Streamlit Community Cloud go to **App settings → Secrets** and add:\n"
            "```toml\nANTHROPIC_API_KEY = \"sk-ant-…\"\n```\n"
            "For local runs, add it to your `.env` file."
        )
        st.stop()
    return True

_check_secrets()


# ── Cached orchestrator (loaded once per worker process) ─────────────────────
# @st.cache_resource means the heavy embedding model and DB connection are
# initialised exactly once, even across multiple user sessions.

@st.cache_resource(show_spinner="🔄 Loading Procurement CoPilot (first run may take ~30s)…")
def _load_orchestrator() -> ProcurementOrchestrator:
    return ProcurementOrchestrator()


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Chat container */
.stChatMessage { border-radius: 12px; margin-bottom: 6px; }

/* Intent badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.badge-structured  { background:#e3f2fd; color:#1565c0; }
.badge-document    { background:#e8f5e9; color:#2e7d32; }
.badge-hybrid      { background:#fff3e0; color:#e65100; }
.badge-out_of_scope{ background:#f3e5f5; color:#6a1b9a; }

/* Citation chips */
.cite-chip {
    display:inline-block;
    background:#f1f8ff;
    border:1px solid #d0e8ff;
    border-radius:12px;
    padding:1px 8px;
    font-size:0.72rem;
    color:#1a73e8;
    margin:2px;
}

/* Sidebar section headers */
.sidebar-section {
    font-size:0.78rem;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:0.08em;
    color:#888;
    margin: 14px 0 4px 0;
}

/* Hide streamlit menu clutter */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────

def _init_state() -> None:
    if "orchestrator" not in st.session_state:
        # Reuse the cached instance — avoids reloading the embedding model
        # every time a user opens a new browser tab.
        st.session_state.orchestrator = _load_orchestrator()
    if "messages" not in st.session_state:
        st.session_state.messages = []          # list of {role, content, meta}
    if "total_questions" not in st.session_state:
        st.session_state.total_questions = 0
    if "intent_counts" not in st.session_state:
        st.session_state.intent_counts = {
            "structured": 0, "document": 0,
            "hybrid": 0, "out_of_scope": 0,
        }


_init_state()


# ── Helper: intent badge HTML ─────────────────────────────────────────────────

INTENT_LABELS = {
    "structured":   ("📊", "STRUCTURED DATA"),
    "document":     ("📄", "CONTRACT ANALYSIS"),
    "hybrid":       ("🔀", "HYBRID"),
    "out_of_scope": ("🚫", "OUT OF SCOPE"),
}

def _intent_badge(intent: str) -> str:
    icon, label = INTENT_LABELS.get(intent, ("❓", intent.upper()))
    return f'<span class="badge badge-{intent}">{icon} {label}</span>'


def _citation_chips(citations: list[str]) -> str:
    chips = "".join(f'<span class="cite-chip">📎 {c}</span>' for c in citations)
    return f'<div style="margin-top:6px">{chips}</div>'


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1170/1170678.png", width=48)
    st.title("Procurement\nCoPilot")
    st.caption("Powered by Claude · Supply Chain Intelligence")

    st.markdown('<div class="sidebar-section">⚡ Quick Queries</div>', unsafe_allow_html=True)

    QUICK_QUERIES = [
        "Top 5 suppliers by annual spend",
        "Suppliers with renewal risk this quarter",
        "Which suppliers have SLA compliance below 85%?",
        "Show POs with price spikes",
        "What are the penalty clauses for late delivery?",
        "What notice period is required to terminate a contract?",
        "Which supplier has the worst SLA — and what does their contract say about penalties?",
        "Unmatched invoices by supplier",
    ]

    for q in QUICK_QUERIES:
        if st.button(q, key=f"quick_{q[:30]}", use_container_width=True):
            st.session_state._pending_question = q

    st.divider()
    st.markdown('<div class="sidebar-section">📈 Session Stats</div>', unsafe_allow_html=True)

    total = st.session_state.total_questions
    counts = st.session_state.intent_counts
    col1, col2 = st.columns(2)
    col1.metric("Total Queries", total)
    col2.metric("Hybrid", counts["hybrid"])
    col1.metric("Structured", counts["structured"])
    col2.metric("Document", counts["document"])

    st.divider()
    st.markdown('<div class="sidebar-section">⚙️ Configuration</div>', unsafe_allow_html=True)

    provider = LLM_PROVIDER.upper()
    model = ANTHROPIC_MODEL if LLM_PROVIDER == "anthropic" else BEDROCK_MODEL_ID
    st.caption(f"**Provider:** {provider}")
    st.caption(f"**Model:** `{model}`")

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.orchestrator.clear_history()
        st.session_state.total_questions = 0
        st.session_state.intent_counts = {k: 0 for k in st.session_state.intent_counts}
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("## 🛒 Procurement CoPilot")
st.markdown(
    "Ask anything about **supplier spend**, **contract terms**, **SLA performance**, "
    "**purchase orders**, or **renewal risk**. I'll route your question to the right data source automatically."
)
st.divider()


# ── Render existing chat history ──────────────────────────────────────────────

def _render_message(msg: dict) -> None:
    """Render a single chat message with all its metadata."""
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            meta = msg.get("meta", {})
            intent = meta.get("intent", "")

            # Intent badge
            if intent:
                st.markdown(_intent_badge(intent), unsafe_allow_html=True)

            # Main answer
            st.markdown(msg["content"])

            # SQL expander (structured / hybrid)
            sql = meta.get("sql_used")
            rows = meta.get("structured_rows") or []
            if sql and sql not in ("CANNOT_ANSWER", None):
                with st.expander(f"🗄️ SQL Query ({len(rows)} rows returned)", expanded=False):
                    st.code(sql, language="sql")
                    if rows:
                        import pandas as pd
                        try:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True)
                        except Exception:
                            for r in rows[:10]:
                                st.write(r)

            # Contract chunks expander (document / hybrid)
            chunks = meta.get("contract_chunks") or []
            if chunks:
                with st.expander(f"📄 Contract Sources ({len(chunks)} excerpts)", expanded=False):
                    for i, chunk in enumerate(chunks, 1):
                        relevance = chunk.get("relevance_pct", "?")
                        src = chunk.get("source_file", "unknown")
                        text = chunk.get("text", "")
                        st.markdown(
                            f"**Excerpt {i}** — `{src}` · relevance: **{relevance}%**"
                        )
                        st.caption(text[:400] + ("…" if len(text) > 400 else ""))
                        if i < len(chunks):
                            st.markdown("---")

            # Citations
            citations = meta.get("citations") or []
            if citations:
                st.markdown(_citation_chips(citations), unsafe_allow_html=True)

        else:
            st.markdown(msg["content"])


for msg in st.session_state.messages:
    _render_message(msg)


# ── Handle pending quick query (set by sidebar buttons) ───────────────────────

pending = st.session_state.pop("_pending_question", None)


# ── Chat input ────────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask about suppliers, contracts, spend, SLA, renewals…")

question = pending or user_input

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Run orchestrator with spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            t0 = time.time()
            try:
                response: OrchestratorResponse = st.session_state.orchestrator.run(question)
                elapsed = time.time() - t0
            except Exception as exc:
                st.error(f"⚠️ Orchestrator error: {exc}")
                st.stop()

        # Update stats
        st.session_state.total_questions += 1
        st.session_state.intent_counts[response.intent] = (
            st.session_state.intent_counts.get(response.intent, 0) + 1
        )

        # Build meta for storage
        meta = {
            "intent": response.intent,
            "sql_used": response.sql_used,
            "structured_rows": response.structured_rows,
            "contract_chunks": response.contract_chunks,
            "citations": response.citations,
            "elapsed_s": round(elapsed, 1),
        }

        # Render immediately
        if response.intent:
            st.markdown(_intent_badge(response.intent), unsafe_allow_html=True)

        st.markdown(response.answer)

        # SQL block
        if response.sql_used and response.sql_used != "CANNOT_ANSWER":
            rows = response.structured_rows or []
            with st.expander(f"🗄️ SQL Query ({len(rows)} rows returned)", expanded=False):
                st.code(response.sql_used, language="sql")
                if rows:
                    import pandas as pd
                    try:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    except Exception:
                        for r in rows[:10]:
                            st.write(r)

        # Contract chunks
        if response.contract_chunks:
            chunks = response.contract_chunks
            with st.expander(f"📄 Contract Sources ({len(chunks)} excerpts)", expanded=False):
                for i, chunk in enumerate(chunks, 1):
                    relevance = chunk.get("relevance_pct", "?")
                    src = chunk.get("source_file", "unknown")
                    text = chunk.get("text", "")
                    st.markdown(
                        f"**Excerpt {i}** — `{src}` · relevance: **{relevance}%**"
                    )
                    st.caption(text[:400] + ("…" if len(text) > 400 else ""))
                    if i < len(chunks):
                        st.markdown("---")

        # Citations + latency
        bottom_cols = st.columns([3, 1])
        with bottom_cols[0]:
            if response.citations:
                st.markdown(_citation_chips(response.citations), unsafe_allow_html=True)
        with bottom_cols[1]:
            st.caption(f"⏱ {elapsed:.1f}s")

    # Store assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "meta": meta,
    })

    # Rerun so sidebar stats refresh
    st.rerun()


# ── Empty state hint ──────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; margin-top:60px; color:#aaa">
        <div style="font-size:3rem">🛒</div>
        <div style="font-size:1.1rem; margin-top:8px">
            Your Procurement Intelligence Assistant is ready.
        </div>
        <div style="font-size:0.85rem; margin-top:4px">
            Use a quick query from the sidebar, or type your own question below.
        </div>
    </div>
    """, unsafe_allow_html=True)
