import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.rag_engine import RAGEngine

@pytest.fixture
def rag():
    engine = RAGEngine(chunk_size=120, chunk_overlap=30)
    return engine

def test_chunking_creates_valid_boundaries(rag):
    sample_text = (
        "Retrieval-Augmented Generation improves LLMs by retrieving relevant documents. "
        "It prevents hallucinations by grounding the output in factual text. "
        "LangChain provides excellent tools for building RAG applications."
    )
    chunks = rag.chunk_text(sample_text, "test_doc.txt")
    assert len(chunks) >= 2
    assert all(c.source == "test_doc.txt" for c in chunks)
    assert all(len(c.text) > 0 for c in chunks)

def test_document_ingestion_indexes_chunks(rag):
    initial_count = len(rag.chunks)
    added = rag.ingest_document("Machine learning requires robust data pipelines.", "doc1.txt")
    assert added >= 1
    assert len(rag.chunks) == initial_count + added

def test_retrieval_returns_relevant_context(rag):
    rag.ingest_document("Python is a popular programming language for AI and backend.", "python.txt")
    rag.ingest_document("Docker is an open-source platform for containerizing applications.", "docker.txt")

    results = rag.retrieve("Tell me about Python for AI", top_k=1)
    assert len(results) == 1
    assert "Python" in results[0]["text"]
    assert results[0]["source"] == "python.txt"

def test_prompt_assembly_enforces_anti_hallucination_guardrails(rag):
    chunks = [{"source": "spec.pdf", "chunk_id": 0, "text": "The maximum temperature is 80C."}]
    prompt = rag.assemble_prompt("What is max temp?", chunks)

    assert "=== CONTEXT ===" in prompt
    assert "spec.pdf" in prompt
    assert "=== USER QUERY ===" in prompt
    assert "What is max temp?" in prompt
    assert "insufficient information" in prompt.lower()

def test_end_to_end_generation(rag):
    rag.ingest_document("StreamSphere uses MERN stack and byte-range video streaming.", "streamsphere.txt")
    res = rag.generate_response("What stack does StreamSphere use?")
    assert "answer" in res
    assert len(res["sources"]) > 0
    assert res["confidence"] > 0.9
