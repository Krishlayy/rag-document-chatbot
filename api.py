from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import sys

sys.path.append(os.path.dirname(__file__))
from core.rag_engine import RAGEngine

app = FastAPI(
    title="RAG-Based Document Chatbot API",
    description="High-performance Retrieval-Augmented Generation REST endpoints",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine()

# Seed with standard system overview
engine.ingest_document(
    "Retrieval-Augmented Generation (RAG) optimizes the output of an LLM by referencing an authoritative knowledge base outside of its training data sources before generating a response. This minimizes hallucinations and guarantees verifiable citations.",
    "RAG_Architecture_Whitepaper.pdf"
)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float

@app.get("/health")
def health_check():
    return {
        "status": "ONLINE",
        "service": "RAG Document Chatbot Engine",
        "indexed_chunks": len(engine.chunks),
    }

@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    result = engine.generate_response(req.query)
    return result

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        text = contents.decode("utf-8")
    except Exception:
        text = str(contents)

    chunks_count = engine.ingest_document(text, file.filename)
    return {
        "filename": file.filename,
        "indexed_chunks": chunks_count,
        "total_chunks_in_store": len(engine.chunks),
    }

@app.get("/documents")
def list_documents():
    sources = list(set([c.source for c in engine.chunks]))
    return {
        "total_documents": len(sources),
        "documents": sources,
        "total_chunks": len(engine.chunks),
    }
