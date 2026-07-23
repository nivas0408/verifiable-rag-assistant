import os
import streamlit as st
import requests
import pandas as pd
from typing import Optional, List, Dict, Any

# Config
import os

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://verifiable-rag-assistant-production.up.railway.app"
)
# Page Configuration
st.set_page_config(
    page_title="Verifiable RAG Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.write(f"Backend URL: {BACKEND_URL}")

# Custom Premium CSS Injection
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
    /* Main App Styles */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .main-title {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Cards and Glassmorphism */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Hover and interactive states */
    .claim-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .supported-badge {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .unsupported-badge {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .contradicted-badge {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Styled source tags */
    .source-tag {
        font-size: 0.75rem;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-family: monospace;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #64748b;
        font-size: 0.85rem;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.3);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to call backend
def get_documents() -> List[Dict[str, Any]]:
    try:
        r = requests.get(
            f"{BACKEND_URL}/documents",
            timeout=15
        )

        if r.status_code == 200:
            return r.json()

    except Exception as e:
        st.error(f"Error connecting to backend: {e}")

    return []

def upload_document(uploaded_file) -> Optional[Dict[str, Any]]:
    try:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        r = requests.post(
            f"{BACKEND_URL}/upload",
            files=files,
            timeout=120
        )

        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Upload failed: {r.json().get('detail', 'Unknown error')}")

    except Exception as e:
        st.error(f"Error uploading file: {e}")

    return None

def delete_document_backend(doc_id: str) -> bool:
    try:
        r = requests.delete(f"{BACKEND_URL}/documents/{doc_id}")
        return r.status_code == 200
    except Exception as e:
        st.error(f"Error deleting document: {e}")
    return False

def query_rag(query: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.post(
            f"{BACKEND_URL}/query",
            json={"query": query},
            timeout=300
        )

        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Query failed: {r.json().get('detail', 'Unknown error')}")

    except Exception as e:
        st.error(f"Error querying backend: {e}")

    return None

# Check backend health & Ollama status
# Check backend health & Ollama status
backend_healthy = False
llm_available = False
llm_provider = "Unknown"

try:
    health_res = requests.get(f"{BACKEND_URL}/health", timeout=15)

    if health_res.status_code == 200:
        backend_healthy = True

        health = health_res.json()

        llm_available = health.get("llm_available", False)

        llm_provider = (
            health.get("metrics", {})
                  .get("settings", {})
                  .get("llm_provider", "Unknown")
        )

except Exception as e:
    st.sidebar.error(f"Health Check Error: {e}")

# Sidebar Layout
with st.sidebar:
    st.markdown("### 🛡️ Knowledge Base Control")
    st.markdown("Upload documents to build your domain-independent verifiable knowledge base.")
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload files (PDF, DOCX, PPTX, TXT, MD, HTML, CSV, JSON)",
        type=["pdf", "docx", "pptx", "txt", "md", "html", "htm", "csv", "json"],
        accept_multiple_files=True,
        key="uploader"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            # We track uploads using session state to avoid double uploads on rerun
            if f"uploaded_{file.name}" not in st.session_state:
                with st.spinner(f"Ingesting {file.name}..."):
                    res = upload_document(file)
                    if res and res.get("status") == "success":
                        st.success(f"Successfully indexed {file.name}")
                        st.session_state[f"uploaded_{file.name}"] = True
                        st.rerun()

    st.markdown("---")
    st.markdown("### 📚 Document Library")
    
    documents = get_documents()
    if documents:
        for doc in documents:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{doc['filename']}**")
                st.caption(f"{doc['file_type'].upper()} • {doc['chunk_count']} chunks • {(doc['file_size']/1024):.1f} KB")
            with col2:
                if st.button("🗑️", key=f"del_{doc['id']}", help=f"Delete {doc['filename']}"):
                    if delete_document_backend(doc['id']):
                        st.success(f"Deleted {doc['filename']}")
                        st.rerun()
    else:
        st.info("No documents uploaded yet.")
        
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    if backend_healthy:
        st.success("Backend: Online")

        if llm_available:
            st.success(f"LLM ({llm_provider}): Connected")
        else:
            st.warning(f"LLM ({llm_provider}): Offline (Fallback Active)")
    else:
        st.error("Backend: Offline")

# Main Content Layout
st.markdown("<h1 class='main-title'>🛡️ Verifiable RAG Assistant</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask domain-independent questions with exact claim-level citation auditing.</div>", unsafe_allow_html=True)

# Initialize Session state for Chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query_result" not in st.session_state:
    st.session_state.last_query_result = None

# Chat History Display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input query
if query_input := st.chat_input("Ask a question about your documents...", disabled=not backend_healthy):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query_input})
    with st.chat_message("user"):
        st.write(query_input)
        
    # Query Backend
    with st.spinner("Retrieving, generating and auditing response..."):
        res = query_rag(query_input)
        if res:
            st.session_state.last_query_result = res
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
            st.rerun()

# Display Detailed Verification Panel for the last query
if st.session_state.last_query_result:
    res = st.session_state.last_query_result
    
    # 2-Column Layout: Column 1 is Assistant Response, Column 2 is Verification Auditor
    col_resp, col_audit = st.columns([5, 5])
    
    with col_resp:
        st.markdown("### 🤖 Assistant Answer")
        
        # Display is_mock indicator if applicable
        if res.get("is_mock"):
            st.warning("⚠️ Local Ollama is offline. Displaying response generated in Mock Mode.")
            
        with st.chat_message("assistant"):
            st.write(res["answer"])
            
        # Display Context Chunks used
        with st.expander("📚 Retrieved Source Chunks"):
            if res.get("chunks"):
                for idx, chunk in enumerate(res["chunks"], 1):
                    meta = chunk["metadata"]

                    st.markdown(
                        f"**[{idx}] {meta.get('source')} (Page {meta.get('page')}, Section '{meta.get('section')}')**"
                    )
                    st.markdown(f"*{chunk['text']}*")

                    rerank_score = chunk.get("rerank_score")

                    if rerank_score is not None:
                        st.caption(f"Cross-Encoder Score: {rerank_score:.4f}")
                    else:
                        st.caption("Cross-Encoder: Disabled")

                    st.markdown("---")
            else:
                st.write("No chunks retrieved.")
                
    with col_audit:
        st.markdown("### 🔍 Citation Verification Auditor")
        
        verification_results = res.get("verification_results", [])
        
        if not verification_results:
            st.info("Ask a question to audit claims and citation grounding.")
        else:
            # 1. Overall grounding statistics
            supported_count = sum(1 for c in verification_results if c["status"] == "SUPPORTED")
            total_claims = len(verification_results)
            grounding_rate = (supported_count / total_claims) * 100 if total_claims > 0 else 0
            
            st.markdown(f"""
            <div class='glass-card'>
                <div style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;'>Grounding Audit Summary</div>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                    <span>Grounded Claims Rate</span>
                    <span style='font-size: 1.5rem; font-weight: 700; color: {"#10b981" if grounding_rate >= 80 else "#f59e0b"};'>{grounding_rate:.1f}%</span>
                </div>
                <div style='color: #94a3b8; font-size: 0.9rem;'>
                    {supported_count} of {total_claims} sentences successfully verified against the source text.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. Render color-coded paragraph
            st.markdown("#### Grounding Heatmap")
            html_answer = "<div style='line-height: 1.8; font-size: 1.05rem; padding: 1rem; border-radius: 8px; background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05);'>"
            
            for claim_data in verification_results:
                claim_text = claim_data["claim"]
                status = claim_data["status"]
                score = claim_data["confidence_score"]
                
                if status == "SUPPORTED":
                    color = "rgba(16, 185, 129, 0.12)"
                    border = "rgba(16, 185, 129, 0.4)"
                    text_color = "#34d399"
                elif status == "CONTRADICTED":
                    color = "rgba(239, 68, 68, 0.12)"
                    border = "rgba(239, 68, 68, 0.4)"
                    text_color = "#f87171"
                else:  # UNSUPPORTED
                    color = "rgba(245, 158, 11, 0.12)"
                    border = "rgba(245, 158, 11, 0.4)"
                    text_color = "#fbbf24"
                    
                html_answer += f'<span style="background-color: {color}; border-bottom: 2px solid {border}; color: {text_color}; padding: 2px 4px; margin: 0 1px; border-radius: 4px;" title="Confidence: {score:.1f}% ({status})">{claim_text}</span>'
                
            html_answer += "</div>"
            st.markdown(html_answer, unsafe_allow_html=True)
            st.caption("ℹ️ Hover over highlighted claims to see confidence score and grounding classification.")
            
            # 3. Interactive Claim-by-Claim breakdown
            st.markdown("#### Claim-by-Claim Grounding Details")
            for idx, claim_data in enumerate(verification_results, 1):
                claim_text = claim_data["claim"]
                status = claim_data["status"]
                score = claim_data["confidence_score"]
                best_cit = claim_data.get("best_citation")
                
                badge_class = "supported-badge" if status == "SUPPORTED" else ("contradicted-badge" if status == "CONTRADICTED" else "unsupported-badge")
                
                with st.expander(f"Sentence {idx}: {claim_text[:60]}..."):
                    st.markdown(f"**Full Claim:** {claim_text}")
                    
                    st.markdown(
                        f"**Status:** <span class='claim-badge {badge_class}'>{status}</span> "
                        f"**Confidence:** `{score:.1f}%`",
                        unsafe_allow_html=True
                    )
                    
                    if best_cit:
                        st.markdown("##### Best Matching Cited Reference")
                        st.markdown(f"📄 **File:** `{best_cit['doc_name']}` • **Page:** `{best_cit['page']}` • **Section:** `{best_cit['section']}`")
                        
                        # Find cited passage from chunks list
                        passage_text = "Cited passage text could not be loaded."
                        source_idx = best_cit["source_idx"] - 1
                        if 0 <= source_idx < len(res["chunks"]):
                            passage_text = res["chunks"][source_idx]["text"]
                            
                        st.info(f"**Verifiable Passage:**\n\n\"{passage_text}\"")
                        similarity = best_cit.get("similarity_score")
                        nli_status = best_cit.get("nli_status", "Unknown")

                        if similarity is not None:
                            st.caption(f"Semantic Alignment: {similarity:.4f} • NLI Check: {nli_status}")
                        else:
                            st.caption(f"Semantic Alignment: N/A • NLI Check: {nli_status}")
                    else:
                        st.warning("No citation marker found in the response for this claim. This statement was produced without explicit grounding.")

st.markdown("<div class='footer'>🛡️ Verifiable RAG System • Created with Streamlit, FastAPI, ChromaDB, and Multi-LLM Support</div>", unsafe_allow_html=True)
