"""
Phase 5 — Streamlit UI
E-Commerce GenAI Pipeline
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag.rag_pipeline import ask

st.set_page_config(
    page_title="Echo Dot Product Intelligence",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Echo Dot Product Intelligence")
st.markdown("Ask any question about the Amazon Echo Dot — answers grounded in **real customer reviews**.")
st.divider()

st.markdown("**Try asking:**")
col1, col2 = st.columns(2)

with col1:
    if st.button("🎵 Good for music?"):
        st.session_state.query = "Is the Echo Dot good for playing music?"
    if st.button("⚙️ Easy to set up?"):
        st.session_state.query = "How easy is it to set up the Echo Dot?"

with col2:
    if st.button("😤 Common complaints?"):
        st.session_state.query = "What are the common complaints about Echo Dot?"
    if st.button("🛏️ Good for bedroom?"):
        st.session_state.query = "Is the sound quality good enough for a bedroom?"

st.divider()

query = st.text_input(
    "Or type your own question:",
    value=st.session_state.get("query", ""),
    placeholder="e.g. Does it work with Spotify?"
)

if st.button("🔍 Ask", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching reviews and generating answer..."):
            try:
                result = ask(query)
                st.divider()
                st.markdown("### 💡 Answer")
                st.success(result["answer"])
                st.markdown(f"### 📋 Based on {len(result['reviews_used'])} reviews")
                for i, review in enumerate(result["reviews_used"]):
                    stars = "⭐" * review["rating"]
                    with st.expander(f"{stars} Review {i+1} — {review['title']}"):
                        st.write(review["text"])
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.divider()
st.caption("Built with PySpark · ChromaDB · sentence-transformers · Groq LLaMA 3.3 · FastAPI · Streamlit")
