#  Procurement CoPilot

An agentic AI assistant for procurement intelligence — powered by **Gemini** and **Streamlit**.

Ask natural-language questions about supplier spend, SLA performance, contract clauses, purchase orders, and renewal risk. The orchestrator automatically routes to a SQL agent, a RAG contract-analysis agent, or both.

---

## Features

- **Intent routing** — classifies every question as STRUCTURED / DOCUMENT / HYBRID / OUT OF SCOPE
- **SQL agent** — translates questions into SQL against a live SQLite procurement database
- **RAG agent** — semantic search over supplier contracts via ChromaDB + sentence-transformers
- **Hybrid synthesis** — both agents run in parallel; Claude synthesises a combined answer
- **Conversation memory** — coreference resolution across multi-turn chats
- **Clean Streamlit UI** — intent badges, SQL expanders, contract source viewers, citation chips

---

## Deploy to Streamlit Community Cloud (Free)

This is the fastest way to get a public URL for your personal branding site.

### Step 1 — Push to GitHub

```bash
# From the project root
git init
git add .
git commit -m "Initial commit: Procurement CoPilot"

# Create a new repo on github.com (e.g. procurement-copilot), then:
git remote add origin https://github.com/<your-username>/procurement-copilot.git
git branch -M main
git push -u origin main
```

> **Note:** The `knowledge_base/` folder (SQLite DB + ChromaDB vector store) **must be committed** — it contains the pre-built synthetic procurement data. The `.gitignore` intentionally does **not** exclude it.

### Step 2 — Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**.
3. Fill in:
   | Field | Value |
   |---|---|
   | Repository | `<your-username>/procurement-copilot` |
   | Branch | `main` |
   | Main file path | `ui/app.py` |
   | App URL (optional) | `procurement-copilot` |
4. Click **"Advanced settings"** and paste your secrets (see below).
5. Click **"Deploy!"** — your app will be live at `https://<your-username>-procurement-copilot.streamlit.app`

### Step 3 — Add secrets

In **App settings → Secrets**, paste:

```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

Get anthropic key at [console.anthropic.com](https://console.anthropic.com).
get API key at (https://ai.google.dev/gemini-api/docs/api-key)

---

## Add to Your Personal Website

Once deployed, you have two options:

### Option A — Link (simplest)

Add a button or card on your portfolio site that links directly to your Streamlit URL:

```html
<a href="https://your-username-procurement-copilot.streamlit.app" target="_blank">
  🛒 Live Demo — Procurement CoPilot
</a>
```

### Option B — Embed via iframe

Embed the app directly into any HTML page:

```html
<iframe
  src="https://your-username-procurement-copilot.streamlit.app?embed=true"
  width="100%"
  height="700"
  style="border: none; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.1);"
  allow="clipboard-write"
></iframe>
```

> The `?embed=true` parameter hides the Streamlit menu bar for a cleaner look.

---

##  Run Locally

```bash
# 1. Clone
git clone https://github.com/<your-username>/procurement-copilot.git
cd procurement-copilot

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 5. Run
streamlit run ui/app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

---

## Project Structure

```
procurement-copilot/
├── ui/
│   └── app.py                  # Streamlit chat interface
├── agents/
│   ├── orchestrator.py         # Intent routing + hybrid synthesis
│   ├── structured_query_agent.py  # NL → SQL agent
│   ├── document_retrieval_agent.py # ChromaDB RAG agent
│   └── llm_client.py           # Anthropic / Bedrock client
├── ingestion/
│   ├── structured_ingester.py  # CSV → SQLite
│   ├── document_ingester.py    # Contracts → ChromaDB
│   └── embedder.py             # sentence-transformers wrapper
├── knowledge_base/
│   ├── structured_db/          # SQLite database (pre-built)
│   └── vector_store/           # ChromaDB (pre-built)
├── data/
│   ├── supplier_master/        # suppliers.csv
│   ├── purchase_orders/        # purchase_orders.csv
│   └── contracts/              # supplier contract .docx files
├── utils/
│   ├── config.py               # Central config (reads Streamlit secrets + .env)
│   └── db_helper.py            # SQLite connection helper
├── synthetic/
│   └── generate_data.py        # Synthetic data generator (dev only)
├── .streamlit/
│   ├── config.toml             # Theme + server settings
│   └── secrets.toml.example    # Secret template
├── requirements.txt
├── packages.txt                # System deps for Streamlit Cloud
└── .gitignore
```

---

## Regenerate Data (optional)

If you want to rebuild the database from scratch with fresh synthetic data:

```bash
python -m synthetic.generate_data
python -m ingestion.structured_ingester
python -m ingestion.document_ingester
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude (claude-sonnet-4) |
| UI | Streamlit |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Structured DB | SQLite via SQLAlchemy |
| Deployment | Streamlit Community Cloud (free) |
