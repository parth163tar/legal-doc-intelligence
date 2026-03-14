import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.qa_chain import answer_query
from models.clause_extractor import extract_clauses_from_text
from models.risk_scorer import score_contract_clauses, generate_risk_report
from retrieval.vector_store import search
import json

app = FastAPI(
    title="Legal Document Intelligence API",
    description="RAG-powered legal contract analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Request/Response Models ────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    n_retrieve: int = 10
    n_rerank: int = 3

class ContractAnalysisRequest(BaseModel):
    contract_text: str
    contract_name: str = "Uploaded Contract"

class SearchRequest(BaseModel):
    query: str
    n_results: int = 5

# ── Routes ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Legal Document Intelligence API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/qa", "/analyze", "/search", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "mistral", "vector_db": "chromadb"}

@app.post("/qa")
def question_answer(req: QueryRequest):
    """Answer a legal question using RAG"""
    try:
        result = answer_query(
            query=req.question,
            n_retrieve=req.n_retrieve,
            n_rerank=req.n_rerank
        )
        return {
            "question": result["query"],
            "answer": result["answer"],
            "num_sources": result["num_sources"],
            "source_chunks": result["source_chunks"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
def analyze_contract(req: ContractAnalysisRequest):
    """Extract clauses and score risk from contract text"""
    try:
        # Extract clauses
        clauses = extract_clauses_from_text(req.contract_text)

        # Score risk
        scored = score_contract_clauses(clauses, use_llm=True)

        # Generate report
        report = generate_risk_report(req.contract_name, scored)

        return {
            "contract": req.contract_name,
            "overall_risk": report["overall_risk"],
            "risk_counts": report["risk_counts"],
            "total_clauses": report["total_clauses"],
            "clauses": report["all_clauses"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def semantic_search(req: SearchRequest):
    """Semantic search across all contracts"""
    try:
        results = search(req.query, n_results=req.n_results)
        return {
            "query": req.query,
            "results": [
                {
                    "text": r["text"],
                    "doc_id": r["metadata"]["doc_id"],
                    "chunk_index": r["metadata"]["chunk_index"]
                }
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))