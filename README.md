# ⚖️ Legal Document Intelligence

> A production-grade RAG system for intelligent legal contract analysis — built on 510 real SEC contracts with semantic search, clause extraction, and AI-powered risk scoring.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red?style=flat-square&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-containerized-blue?style=flat-square&logo=docker)
![Railway](https://img.shields.io/badge/Deployed-Railway-purple?style=flat-square)

---

## 🎯 What It Does

Ask plain English questions about 510 real legal contracts and get grounded, cited answers in seconds.

```
"What is the liability cap in these contracts?"
→ "The liability cap shall not exceed $1,000,000 as stated in Section 11.3."

"What are the indemnification obligations between parties?"
→ "Each party shall indemnify and hold harmless the other from
   third-party claims arising from breach." (Section 8)
```

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Streamlit UI  ──────────►  FastAPI Backend (Railway)
                                    │
                          ┌─────────┼─────────┐
                          ▼         ▼         ▼
                      pgvector   Cohere    Groq LLM
                    PostgreSQL  Reranker  Llama 3.1
                    (80,532     (top 10   (generates
                     chunks)   → top 3)   answer)
```

### Pipeline: Retrieve → Rerank → Generate

| Stage | Technology | Purpose |
|-------|-----------|---------|
| **Embedding** | all-MiniLM-L6-v2 (384-dim) | Convert text to vectors |
| **Storage** | PostgreSQL + pgvector | HNSW vector similarity search |
| **Retrieval** | Cosine similarity ANN | Top-10 relevant chunks |
| **Reranking** | Cohere rerank-english-v3.0 | Refine to top-3 |
| **Generation** | Groq Llama-3.1-8b-instant | Grounded answer with citations |

---

## 📊 Dataset

- **Source:** CUAD (Contract Understanding Atticus Dataset)
- **510 real contracts** from SEC EDGAR public filings
- **80,532 chunks** @ 512 tokens with 64-token overlap
- **41 labeled clause types** including liability, termination, IP, non-compete

---

## ✨ Features

### 💬 Contract Q&A
Ask any legal question across all 510 contracts. Answers are grounded in actual contract text with source citations.

### 📄 Clause Extraction
Automatically identifies 10 key clause types from any pasted contract:
- Liability Cap, Termination, IP Ownership, Non-Compete
- Confidentiality, Indemnification, Auto-Renewal, Assignment
- Dispute Resolution, Governing Law

### ⚠️ Risk Scoring
Each clause scored LOW / MEDIUM / HIGH using a hybrid approach:
- **Rule-based:** Fast keyword matching for known risk signals
- **LLM-based:** Deep semantic analysis with red flag identification

### 🔍 Semantic Search
Search across all contracts using natural language — finds relevant clauses even without exact keyword matches.

---

## 📈 Evaluation Results

Custom LLM-as-judge evaluation framework (built because RAGAS timed out with local models):

| Metric | Score | Target |
|--------|-------|--------|
| Answer Relevancy | **1.000** | 0.80 |
| Faithfulness | **0.960** | 0.85 |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11
- Ollama (for local LLM)
- PostgreSQL or ChromaDB

### Local Setup

```bash
# Clone the repository
git clone https://github.com/parth163tar/legal-doc-intelligence.git
cd legal-doc-intelligence

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

```env
GROQ_API_KEY=gsk_...
COHERE_API_KEY=...
DATABASE_URL=postgresql://...  # Optional, uses ChromaDB if not set
CHROMA_DB_PATH=./chroma_db
```

### Run Locally

```bash
# Terminal 1 — Start API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Start UI
streamlit run ui/app.py

# Optional — Start local LLM
ollama serve
ollama pull mistral
```

Open `http://localhost:8501` in your browser.

### Ingest Data

```bash
# Download CUAD dataset
# Place CUADv1.json in data/ folder

# Ingest all 510 contracts
python ingestion/embedder.py

# Or ingest to PostgreSQL
python ingest_to_postgres.py
```

---

## 🌐 Live Demo

| Service | URL |
|---------|-----|
| **API** | `https://legal-doc-intelligence-production.up.railway.app` |
| **API Docs** | `https://legal-doc-intelligence-production.up.railway.app/docs` |
| **UI** | `https://legaldocintelligence.streamlit.app/` |

### API Endpoints

```bash
# Health check
GET /health

# Ask a legal question
POST /qa
{"question": "What is the liability cap?", "n_retrieve": 10, "n_rerank": 3}

# Analyze a contract
POST /analyze
{"contract_text": "...", "contract_name": "My Contract"}

# Semantic search
POST /search
{"query": "termination clause", "n_results": 5}
```

---

## 📁 Project Structure

```
legal-doc-intelligence/
├── api/
│   └── main.py              # FastAPI REST API
├── ingestion/
│   ├── chunker.py           # RecursiveCharacterTextSplitter
│   └── embedder.py          # Batch embedding pipeline
├── retrieval/
│   └── vector_store.py      # PostgreSQL/ChromaDB vector search
├── models/
│   ├── qa_chain.py          # RAG pipeline (retrieve→rerank→generate)
│   ├── clause_extractor.py  # LLM-based clause extraction
│   └── risk_scorer.py       # Hybrid rule+LLM risk scoring
├── evaluation/
│   └── ragas_eval.py        # Custom LLM-as-judge evaluation
├── ui/
│   └── app.py               # Streamlit web interface
├── data/                    # CUAD dataset (not in repo)
├── chroma_db/               # Local vector DB (not in repo)
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-service orchestration
└── requirements.txt
```

---

## 🐳 Docker

```bash
# Build and run all services
docker compose up --build

# Services:
# - legal-ollama  → port 11434 (local LLM)
# - legal-api     → port 8000  (FastAPI)
# - legal-ui      → port 8501  (Streamlit)
```

---

## 🧠 Technical Decisions

| Decision | Rationale |
|----------|-----------|
| ChromaDB → pgvector | Persistent cloud storage, no re-embedding on restart |
| Local Mistral → Groq | 10x faster on cloud, no GPU needed |
| RAGAS → Custom evaluator | RAGAS timed out with local LLM; LLM-as-judge is a valid production pattern |
| 512 chunk size | Matches all-MiniLM-L6-v2 max input length |
| 64 token overlap | Prevents clause boundary loss at chunk edges |
| Batch size 100 | Balances memory usage vs ingestion speed |

---

## 🔮 Future Improvements

- [ ] Fine-tune embedding model on legal domain
- [ ] Add PDF upload for custom contract analysis
- [ ] Cross-document comparison queries
- [ ] Structured clause database for comparative analysis
- [ ] Authentication and multi-user support
- [ ] Contract summarization endpoint

---

## 📦 Tech Stack

**AI/ML:** LangChain · Groq (Llama 3.1) · Cohere · sentence-transformers · RAGAS

**Backend:** FastAPI · PostgreSQL · pgvector · ChromaDB · Python 3.11

**Frontend:** Streamlit

**Infrastructure:** Docker · Railway · Streamlit Cloud

**Data:** CUAD Dataset (510 SEC contracts)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built with ⚖️ for legal intelligence</p>
