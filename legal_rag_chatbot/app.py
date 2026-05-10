# ============================================================
#  app.py — Streamlit Frontend (run with: streamlit run app.py)
# ============================================================

import os
import time
import streamlit as st
from rag_backend import get_collections, build_rag_chain, ask_question, summarize_collection

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Pakistan Legal AI",
    page_icon  = "⚖️",
    layout     = "wide"
)

# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS — Animations, Typography, and Glass Effects
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 20px 0 10px 0;
        color: #38bdf8;
        font-weight: 800;
        font-size: 3.5rem;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }
    
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 40px;
        animation: fadeInUp 0.8s ease-out;
        font-weight: 300;
    }

    /* Citation badge with glassmorphism */
    .citation-box {
        background: rgba(56, 189, 248, 0.1);
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 13px;
        color: #e2e8f0;
        margin-top: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    
    .citation-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(56, 189, 248, 0.15);
        background: rgba(56, 189, 248, 0.15);
    }

    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Button styling upgrades */
    .stButton>button {
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
    }

    /* Upload Box styling */
    .upload-prompt {
        text-align: center;
        padding: 50px;
        border: 2px dashed #334155;
        border-radius: 20px;
        background: rgba(30, 41, 59, 0.5);
        margin: 40px auto;
        max-width: 600px;
    }
    .upload-prompt h3 { color: #f8fafc; }
    .upload-prompt p { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">⚖️ Pakistan Legal AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Groq LLaMA3 · Hybrid Search · Active Reranking</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  SIDEBAR — Document info & controls
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 4rem; margin-bottom: 0;'>🏛️</h1>", unsafe_allow_html=True)
    st.markdown("### ⚙️ System Config")

    collections = get_collections()
    selected_collection = st.selectbox(
        "📂 Document Collection",
        collections,
        index=0
    )
    
    # If the collection changed, reset chat and chain
    if "current_collection" not in st.session_state or st.session_state.current_collection != selected_collection:
        st.session_state.current_collection = selected_collection
        st.session_state.pop("rag_chain", None)
        st.session_state.chat_history = []

    st.info("""
    **Model:** LLaMA-3.1-8B (Groq)  
    **Search:** Hybrid (FAISS + BM25)  
    **Reranker:** ms-marco-MiniLM
    """)

    st.markdown("---")

    # Summarization Button
    if st.button("📝 Generate Summary", use_container_width=True):
        with st.spinner("Analyzing documents to generate an executive summary..."):
            summary = summarize_collection(selected_collection)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"**Executive Summary of '{selected_collection}'**:\n\n{summary}",
                "citations": []
            })
            st.rerun()

    # Clear chat history
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.pop("rag_chain", None)
        st.success("Chat cleared!")

    st.markdown("---")
    st.markdown("**Try asking:**")
    sample_questions = [
        "What are the termination clauses?",
        "What is the probationary period?",
        "Explain the confidentiality obligations.",
        "What happens in case of a dispute?",
    ]
    for q in sample_questions:
        if st.button(f"💬 {q}", use_container_width=True):
            st.session_state.sample_input = q


# ─────────────────────────────────────────────────────────────
#  LOAD RAG CHAIN
# ─────────────────────────────────────────────────────────────
if "rag_chain" not in st.session_state:
    with st.spinner(f"🚀 Initializing Neural Core for '{selected_collection}'..."):
        try:
            st.session_state.rag_chain = build_rag_chain(selected_collection)
        except ValueError as e:
            # Graceful empty state UI
            st.markdown(f"""
            <div class="upload-prompt">
                <h1 style='font-size: 3rem;'>📂</h1>
                <h3>No Documents Found</h3>
                <p>Please place your PDF documents inside the <b>documents/{selected_collection}/</b> folder and refresh the page to start the AI engine.</p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()


# ─────────────────────────────────────────────────────────────
#  CHAT HISTORY INIT
# ─────────────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.session_state.chat_history.append({
        "role"    : "assistant",
        "content" : (
            f"Hello! I am your **Pakistan Legal AI Assistant**. \n\n"
            f"I have successfully loaded the **{selected_collection}** documents into my active memory. "
            "Every answer I provide is backed by Hybrid Search and active Reranking to completely prevent hallucinations. "
            "\n\nHow can I assist you with your legal documents today?"
        ),
        "citations": []
    })


# ─────────────────────────────────────────────────────────────
#  HELPER: STREAMING EFFECT
# ─────────────────────────────────────────────────────────────
def stream_data(text):
    """Simulates a typing effect for the LLM response."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)


# ─────────────────────────────────────────────────────────────
#  DISPLAY CHAT HISTORY
# ─────────────────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.chat_history):
    avatar = "🧑‍💼" if msg["role"] == "user" else "⚖️"
    
    with st.chat_message(msg["role"], avatar=avatar):
        # Only stream the VERY LAST assistant message if it was just added
        if msg["role"] == "assistant" and idx == len(st.session_state.chat_history) - 1 and st.session_state.get("is_new_message"):
            st.write_stream(stream_data(msg["content"]))
            st.session_state.is_new_message = False  # Turn off streaming after it runs once
        else:
            st.markdown(msg["content"])
            
        # Display Citations
        if msg.get("citations"):
            citation_html = "".join([
                f'📄 <b>{c["source"]}</b> &nbsp; | &nbsp; Page {c["page"]}<br>'
                for c in msg["citations"]
            ])
            st.markdown(
                f'<div class="citation-box">📌 <b>Verified Sources:</b><br>{citation_html}</div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────
#  CHAT INPUT
# ─────────────────────────────────────────────────────────────
# Handle sample question click from sidebar
default_input = st.session_state.pop("sample_input", "")

user_input = st.chat_input("Ask a legal question... e.g. What are the termination clauses?")

if not user_input and default_input:
    user_input = default_input


# ─────────────────────────────────────────────────────────────
#  PROCESS USER QUESTION
# ─────────────────────────────────────────────────────────────
if user_input:
    # 1. Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "citations": []
    })
    
    # Rerun immediately to show user message before fetching answer
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(user_input)

    # 2. Get answer from RAG chain
    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("🔍 Deep-scanning documents..."):
            answer, citations = ask_question(
                st.session_state.rag_chain,
                user_input
            )
            
        # 3. Stream the answer live
        st.write_stream(stream_data(answer))
        
        # 4. Show citations
        if citations:
            citation_html = "".join([
                f'📄 <b>{c["source"]}</b> &nbsp; | &nbsp; Page {c["page"]}<br>'
                for c in citations
            ])
            st.markdown(
                f'<div class="citation-box">📌 <b>Verified Sources:</b><br>{citation_html}</div>',
                unsafe_allow_html=True
            )
            
    # 5. Append to history so it stays on next reload
    st.session_state.is_new_message = False # Already streamed it
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "citations": citations
    })
