import sys

import os

# -------- FIX PATH (for Streamlit Cloud) --------

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from app.agent import run_agent

# -------- PAGE CONFIG --------

st.set_page_config(page_title="Eco AI Agent", layout="wide")

# -------- HEADER --------

st.title("🌍 Eco AI Assistant")

st.caption("RAG + Tool Calling Agent")

# -------- SESSION MEMORY --------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------- DISPLAY CHAT --------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------- USER INPUT --------

user_input = st.chat_input("Ask about water, pollution, environment...")

if user_input:

    # Save user message

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):

        st.markdown(user_input)

    # -------- AI RESPONSE --------

    with st.chat_message("assistant"):

        with st.spinner("Thinking... 🤖"):
            response = run_agent(st.session_state.messages)

            st.markdown(response)

            # -------- TOOL INDICATOR --------

            if any(word in user_input.lower() for word in ["air", "aqi", "pollution"]):
                st.caption("⚙️ Used real-time API data")

    # Save response

    st.session_state.messages.append({"role": "assistant", "content": response})
