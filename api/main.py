"""
Phase 4 — FastAPI Serving Layer
E-Commerce GenAI Pipeline

Wraps the RAG pipeline in a REST endpoint.
Run: uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add project root to path so we can import rag module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.rag_pipeline import ask

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="E-Commerce Product Intelligence API",
    description="RAG-powered product Q&A using real customer reviews",
    version="1.0.0"
)

# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

class ReviewUsed(BaseModel):
    id: str
    rating: int
    title: str
    text: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    reviews_used: list[ReviewUsed]

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "E-Commerce Product Intelligence API",
        "usage": "POST /ask with {query: your question}",
        "docs": "http://localhost:8000/docs"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = ask(request.query)
        return QueryResponse(
            question = result["question"],
            answer   = result["answer"],
            reviews_used = [
                ReviewUsed(
                    id     = r["id"],
                    rating = r["rating"],
                    title  = r["title"],
                    text   = r["text"]
                )
                for r in result["reviews_used"]
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
