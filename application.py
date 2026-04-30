import sys
import os
import streamlit as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.aquaiq_ai.agent import WaterAgent
st.set_page_config(
   page_title="Water Treatment Assistant",
   layout="wide"
)
# dark mode - i like black backgrounds
st.markdown("""
<style>
   .stApp {
       background-color: #0e1117;
       color: #fafafa;
   }
   .stChatMessage {
       background-color: #1e1e2e;
       border-radius: 10px;
       padding: 10px;
       margin: 5px 0;
   }
   h1, h2, h3 {
       color: #4a9eff;
   }
</style>
""", unsafe_allow_html=True)
st.title("Water Treatment Assistant")
st.caption("Ask about water treatment or water quality")
# checking env vars
required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
missing = [v for v in required_vars if not os.getenv(v)]
if missing:
   st.error(f"Missing: {missing}")
   st.stop()
# init session
if "agent" not in st.session_state:
   with st.spinner("Loading..."):
       try:
           st.session_state.agent = WaterAgent()
           st.session_state.messages = []
       except Exception as e:
           st.error(f"Start failed: {e}")
           st.stop()
# sidebar
with st.sidebar:
   st.markdown("### About")
   st.markdown("""
   Handles:
   1. PDF questions
   2. Water quality API
   3. Both together
   """)
   st.markdown("### Counties")
   st.markdown("""
   - Travis County, Texas
   - Williamson County, Texas
   - Benton County, Arkansas
   - Baxter County, Arkansas
   - Prince George County, Maryland
   - Oklahoma County, Oklahoma
   """)
   if st.button("Clear"):
       st.session_state.agent.reset()
       st.session_state.messages = []
       st.rerun()
# show history
for msg in st.session_state.messages:
   with st.chat_message(msg["role"]):
       st.markdown(msg["content"])
# user input
if prompt := st.chat_input("Ask something..."):
   st.session_state.messages.append({"role": "user", "content": prompt})
   with st.chat_message("user"):
       st.markdown(prompt)
   with st.chat_message("assistant"):
       with st.spinner("wait..."):
           try:
               response = st.session_state.agent.chat(prompt)
               st.markdown(response)
           except Exception as e:
               st.error(f"Error: {e}")
               response = str(e)
   st.session_state.messages.append({"role": "assistant", "content": response})