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
@app.on_event("startup")
async def startup_event():
    """Load demo data into ChromaDB on startup"""
    from retrieval.vector_store import add_documents, collection
    
    # Check if already populated
    if collection.count() > 0:
        print(f"✅ ChromaDB already has {collection.count()} chunks")
        return
    
    print("📄 Loading demo contracts into ChromaDB...")
    demo_chunks = [
        {"id": "demo_1", "text": "The liability cap shall not exceed $1,000,000. Section 11.3 Liability Cap.", "doc_id": "DEMO_CONTRACT", "chunk_index": 0},
        {"id": "demo_2", "text": "Either party may terminate this Agreement upon 30 days written notice. Upon termination all licenses cease.", "doc_id": "DEMO_CONTRACT", "chunk_index": 1},
        {"id": "demo_3", "text": "Each party retains ownership of intellectual property rights owned prior to this Agreement. IP Ownership Section 4.1.", "doc_id": "DEMO_CONTRACT", "chunk_index": 2},
        {"id": "demo_4", "text": "This Agreement shall be governed by the laws of the State of Delaware. Governing Law Section 12.", "doc_id": "DEMO_CONTRACT", "chunk_index": 3},
        {"id": "demo_5", "text": "Each party shall indemnify and hold harmless the other from third-party claims arising from breach. Indemnification Section 8.", "doc_id": "DEMO_CONTRACT", "chunk_index": 4},
        {"id": "demo_6", "text": "Distributor agrees not to engage in competitive business for 2 years after termination. Non-Compete Section 6.", "doc_id": "DEMO_CONTRACT", "chunk_index": 5},
        {"id": "demo_7", "text": "This Agreement shall automatically renew for one year terms unless terminated with 60 days notice. Auto-Renewal Section 3.", "doc_id": "DEMO_CONTRACT", "chunk_index": 6},
        {"id": "demo_8", "text": "Neither party may assign this Agreement without prior written consent of the other party. Assignment Section 9.", "doc_id": "DEMO_CONTRACT", "chunk_index": 7},
        {"id": "demo_9", "text": "All disputes shall be resolved through binding arbitration in Delaware. Dispute Resolution Section 13.", "doc_id": "DEMO_CONTRACT", "chunk_index": 8},
        {"id": "demo_10", "text": "Confidential information shall not be disclosed to third parties for 5 years after termination. Confidentiality Section 7.", "doc_id": "DEMO_CONTRACT", "chunk_index": 9},
    ]
    add_documents(demo_chunks)
    print(f"✅ Loaded {len(demo_chunks)} demo chunks into ChromaDB")

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