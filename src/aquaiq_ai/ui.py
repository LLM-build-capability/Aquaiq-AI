import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.aquaiq_ai.agent import run_agent

# -------- PAGE CONFIG --------
st.set_page_config(page_title="Eco AI Assistant", layout="wide")

# -------- HEADER --------
st.title("Eco AI Assistant")
st.caption("RAG + Tool-Using Agent (Azure OpenAI)")

# -------- SESSION STATE --------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------- DISPLAY CHAT --------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------- INPUT --------
user_input = st.chat_input("Ask about water, pollution, environment...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # -------- RESPONSE --------
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                response = run_agent(st.session_state.messages)
                st.markdown(response)

                if any(word in user_input.lower() for word in ["air", "aqi", "pollution"]):
                    st.caption("Used real-time API data")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.session_state.messages.append(
        {"role": "assistant", "content": response if 'response' in locals() else "Error occurred"})