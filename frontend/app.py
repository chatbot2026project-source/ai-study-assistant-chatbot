import streamlit as st
import sys
import os

# ---- FORCE BACKEND INTO PYTHON PATH (STREAMLIT SAFE) ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.chatbot_model import get_response


st.set_page_config(page_title="AI Study Assistant", page_icon="📘")

st.title("📘 AI Study Assistant Chatbot")
st.write("Ask your study-related questions below.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Your question:")

if st.button("Ask"):
    if user_input.strip():
        reply = get_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", reply))

for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")
