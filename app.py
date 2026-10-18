import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(__file__))
from core.rag_engine import RAGEngine

st.set_page_config(
    page_title="RAG-Based Document Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Initialize Session State
if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine()
    # Ingest baseline documentation
    st.session_state.engine.ingest_document(
        "Retrieval-Augmented Generation (RAG) is an AI framework that retrieves relevant facts from an external knowledge base to ground Large Language Models on accurate, up-to-date information.",
        "System_Overview.pdf"
    )
    st.session_state.engine.ingest_document(
        "LangChain is a framework designed to simplify the creation of applications using large language models. It provides chains, memory, and vector store integrations.",
        "LangChain_Manual.pdf"
    )

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your RAG Document Assistant. Upload documents on the sidebar or ask questions about existing indexed knowledge."}
    ]

# Sidebar
with st.sidebar:
    st.title("📚 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Documents (TXT, MD, PDF)",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for file in uploaded_files:
            try:
                content = file.read().decode("utf-8")
                chunks = st.session_state.engine.ingest_document(content, file.name)
                st.success(f"Indexed **{file.name}** ({chunks} chunks)")
            except Exception as e:
                st.error(f"Error parsing {file.name}: {e}")

    st.markdown("---")
    st.markdown("### 📊 Index Statistics")
    st.metric("Total Indexed Chunks", len(st.session_state.engine.chunks))
    st.markdown("**Vector Store:** ChromaDB / Semantic Embeddings")
    st.markdown("**Framework:** LangChain & Python")

# Main Chat Interface
st.title("🤖 RAG-Based Document Chatbot")
st.caption("Context-Aware Semantic Search & Hallucination-Grounded Generation")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = st.session_state.engine.generate_response(prompt)
    answer_text = response["answer"]

    with st.chat_message("assistant"):
        st.markdown(answer_text)
        if response.get("sources"):
            with st.expander("🔍 View Retrieved Source Citations"):
                for src in response["sources"]:
                    st.markdown(f"**Source:** `{src['source']}` (Chunk #{src['chunk_id']})")
                    st.info(src["text"])

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
