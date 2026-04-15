"""Streamlit UI for the Capital Finder Bot."""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Capital Finder Bot", page_icon="🌍")
st.title("Capital Finder Bot")
st.caption("Ask me about any country's capital city!")

# Check for FAISS index
if not os.path.exists("faiss_index"):
    st.error(
        "FAISS index not found. Run `python build_vectorstore.py` first to build the index."
    )
    st.stop()

# Load chain once per session
if "chain" not in st.session_state:
    from rag_chain import create_chain
    st.session_state.chain = create_chain()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("e.g. What is the capital of Japan?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = st.session_state.chain.invoke(question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
