import os
import streamlit as st
from dotenv import load_dotenv

from pdf_processor import extract_and_split_pdf
from vector_store import build_vectorstore_from_chunks, load_existing_vectorstore
from rag_engine import query_rag
from ai_engine import query_ai

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Umer PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css(file_name: str = "style.css"):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_mode" not in st.session_state:
    st.session_state.active_mode = "RAG"  # 'RAG' or 'AI'

if "vector_store" not in st.session_state:
    # Try loading existing Chroma_DB if available
    st.session_state.vector_store = load_existing_vectorstore()

if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None

if "last_uploaded_file_id" not in st.session_state:
    st.session_state.last_uploaded_file_id = None

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="margin: 0; color: #f8fafc; font-weight: 700; letter-spacing: -0.5px;">
                📄 Umer PDF Assistant
            </h2>
            <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">
                RAG Engine & AI Intelligence
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Mode Switcher
    st.markdown("<div class='sidebar-title'>⚡ Assistant Mode</div>", unsafe_allow_html=True)
    col_mode1, col_mode2 = st.columns(2)
    with col_mode1:
        if st.button(
            "📄 RAG Mode",
            use_container_width=True,
            type="primary" if st.session_state.active_mode == "RAG" else "secondary"
        ):
            st.session_state.active_mode = "RAG"
            st.rerun()

    with col_mode2:
        if st.button(
            "🤖 AI Mode",
            use_container_width=True,
            type="primary" if st.session_state.active_mode == "AI" else "secondary"
        ):
            st.session_state.active_mode = "AI"
            st.rerun()

    # Active mode explanation banner
    if st.session_state.active_mode == "RAG":
        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 10px 12px; margin-bottom: 18px; font-size: 0.82rem; color: #6ee7b7;">
                <strong>📄 RAG Mode Active:</strong> Answers are strictly grounded in your uploaded PDF document.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 10px; padding: 10px 12px; margin-bottom: 18px; font-size: 0.82rem; color: #c084fc;">
                <strong>🤖 AI Mode Active:</strong> Answers from general AI intelligence without document restrictions.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 2. PDF Document Upload
    st.markdown("<div class='sidebar-title'>📂 Upload Any PDF</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF to query it using RAG mode",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        file_signature = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.last_uploaded_file_id != file_signature:
            with st.spinner("🔄 Processing & indexing PDF..."):
                try:
                    chunks, stats = extract_and_split_pdf(uploaded_file)
                    # Build Chroma vector store
                    vs = build_vectorstore_from_chunks(chunks)
                    st.session_state.vector_store = vs
                    st.session_state.doc_stats = stats
                    st.session_state.last_uploaded_file_id = file_signature
                    st.toast(f"✅ Successfully indexed {stats['filename']} ({stats['total_chunks']} chunks)!", icon="🎉")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

    # 3. Document Statistics Card
    if st.session_state.doc_stats:
        stats = st.session_state.doc_stats
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div style="font-weight: 600; color: #f1f5f9; font-size: 0.9rem; word-break: break-all; margin-bottom: 6px;">
                    📑 {stats['filename']}
                </div>
                <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-size: 0.75rem; padding: 2px 8px; border-radius: 12px; font-weight: 600;">
                    ✓ Indexed & Ready
                </span>
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-val">{stats['total_pages']}</div>
                        <div class="stat-label">Pages</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val">{stats['total_chunks']}</div>
                        <div class="stat-label">Chunks</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif st.session_state.vector_store:
        st.markdown(
            """
            <div class="sidebar-card">
                <div style="font-weight: 600; color: #f1f5f9; font-size: 0.9rem;">
                    📑 Default Vector Store
                </div>
                <span style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; font-size: 0.75rem; padding: 2px 8px; border-radius: 12px; font-weight: 600;">
                    ✓ Chroma_DB Loaded
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="padding: 12px; border-radius: 8px; border: 1px dashed rgba(255, 255, 255, 0.15); text-align: center; color: #94a3b8; font-size: 0.85rem;">
                No PDF currently loaded.<br/>Upload a PDF above to use RAG mode.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 4. Actions
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col_act2:
        if st.button("🔄 Reset Doc", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.doc_stats = None
            st.session_state.last_uploaded_file_id = None
            st.rerun()

    # Footer
    st.markdown(
        """
        <div style="text-align: center; margin-top: 30px; font-size: 0.75rem; color: #64748b;">
            Umer PDF Assistant • Mistral AI & ChromaDB
        </div>
        """,
        unsafe_allow_html=True
    )


# --- MAIN CHAT AREA ---

# Header Bar
active_badge_html = (
    '<div class="mode-badge-rag">📄 RAG Mode (Document Grounded)</div>'
    if st.session_state.active_mode == "RAG"
    else '<div class="mode-badge-ai">🤖 AI Mode (General Knowledge)</div>'
)

st.markdown(
    f"""
    <div class="main-header">
        <div class="header-title-box">
            <h1>Umer PDF Assistant</h1>
            <div class="header-subtitle">
                Intelligent Document Q&A and Conversational AI System
            </div>
        </div>
        <div>
            {active_badge_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Welcome State if no messages
if not st.session_state.messages:
    if st.session_state.active_mode == "RAG":
        st.markdown(
            """
            <div style="background: rgba(255, 255, 255, 0.02); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 35px 25px; text-align: center; margin: 30px 0;">
                <div style="font-size: 2.5rem; margin-bottom: 10px;">📄</div>
                <h3 style="color: #f1f5f9; margin-bottom: 8px;">Welcome to RAG Document Mode</h3>
                <p style="color: #94a3b8; max-width: 550px; margin: 0 auto 16px auto; font-size: 0.95rem;">
                    Upload any PDF using the sidebar and ask questions. The assistant will answer strictly from the document context.
                </p>
                <div style="display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;">
                    <span style="background: rgba(255, 255, 255, 0.05); padding: 6px 14px; border-radius: 20px; font-size: 0.82rem; color: #cbd5e1;">💡 "Summarize the key points"</span>
                    <span style="background: rgba(255, 255, 255, 0.05); padding: 6px 14px; border-radius: 20px; font-size: 0.82rem; color: #cbd5e1;">💡 "What does the document say about..."</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(255, 255, 255, 0.02); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 35px 25px; text-align: center; margin: 30px 0;">
                <div style="font-size: 2.5rem; margin-bottom: 10px;">🤖</div>
                <h3 style="color: #f1f5f9; margin-bottom: 8px;">Welcome to AI Knowledge Mode</h3>
                <p style="color: #94a3b8; max-width: 550px; margin: 0 auto 16px auto; font-size: 0.95rem;">
                    Ask any general knowledge question, request code, explanations, or creative ideas without document constraints.
                </p>
                <div style="display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;">
                    <span style="background: rgba(255, 255, 255, 0.05); padding: 6px 14px; border-radius: 20px; font-size: 0.82rem; color: #cbd5e1;">💡 "Explain how quantum computing works"</span>
                    <span style="background: rgba(255, 255, 255, 0.05); padding: 6px 14px; border-radius: 20px; font-size: 0.82rem; color: #cbd5e1;">💡 "Write a Python script for..."</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Display Chat Messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div class="chat-msg user-msg">
                <div class="msg-header">
                    <div class="msg-sender">👤 You</div>
                </div>
                <div>{msg["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif msg["role"] == "assistant":
        mode_tag = (
            '<span style="color: #34d399; font-size: 0.75rem; background: rgba(16, 185, 129, 0.15); padding: 2px 8px; border-radius: 10px; font-weight: 600;">📄 RAG</span>'
            if msg.get("mode") == "RAG"
            else '<span style="color: #c084fc; font-size: 0.75rem; background: rgba(139, 92, 246, 0.15); padding: 2px 8px; border-radius: 10px; font-weight: 600;">🤖 AI Mode</span>'
        )
        
        st.markdown(
            f"""
            <div class="chat-msg assistant-msg">
                <div class="msg-header">
                    <div class="msg-sender">✨ Umer PDF Assistant &nbsp;{mode_tag}</div>
                </div>
                <div style="color: #f1f5f9;">{msg["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Show expandable sources if available in RAG mode
        if msg.get("sources"):
            with st.expander(f"📑 View Retrieved Sources ({len(msg['sources'])} passages)", expanded=False):
                for idx, doc in enumerate(msg["sources"], 1):
                    page_num = doc.metadata.get("page", 0) + 1 if "page" in doc.metadata else "N/A"
                    source_name = doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", "Document"))
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <strong style="color: #38bdf8;">[Source #{idx}] - {source_name} (Page {page_num})</strong><br/>
                            <div style="margin-top: 5px; color: #cbd5e1; white-space: pre-wrap;">{doc.page_content.strip()}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# User Chat Input
query = st.chat_input("Ask a question about your document or anything else...")

if query:
    # Append user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    current_mode = st.session_state.active_mode

    # Process query based on active mode
    if current_mode == "RAG":
        if not st.session_state.vector_store:
            st.warning("⚠️ No document is currently indexed for RAG. Please upload a PDF in the sidebar or switch to AI Mode.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⚠️ Please upload a PDF document in the sidebar to use RAG mode, or switch to **🤖 AI Mode** to chat freely.",
                "mode": "RAG",
                "sources": []
            })
            st.rerun()
        else:
            with st.spinner("🔍 Retrieving from document and generating answer..."):
                try:
                    result = query_rag(query, st.session_state.vector_store)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "mode": "RAG",
                        "sources": result["docs"]
                    })
                except Exception as e:
                    st.error(f"Error during RAG query: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"An error occurred while generating response: {str(e)}",
                        "mode": "RAG",
                        "sources": []
                    })
            st.rerun()

    elif current_mode == "AI":
        with st.spinner("🤖 Thinking..."):
            try:
                answer = query_ai(query, chat_history=st.session_state.messages[:-1])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "mode": "AI",
                    "sources": []
                })
            except Exception as e:
                st.error(f"Error during AI query: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"An error occurred while generating response: {str(e)}",
                    "mode": "AI",
                    "sources": []
                })
        st.rerun()
