import streamlit as st
import sys
import os

# Backend path setup
BACKEND_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "backend")
)
sys.path.append(BACKEND_PATH)

from chatbot_model import get_response

st.set_page_config(page_title="AI Study Assistant", page_icon="📘")

st.title("📘 AI Study Assistant Chatbot")
st.write("Ask your study-related questions below.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""


user_input = st.text_input("Your question:")

if st.button("Ask"):
    if user_input.strip():

        # remember last topic
        st.session_state.last_topic = user_input

        bot_reply = get_response(user_input)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", bot_reply))
# 
# Display chat history
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 Bot:** {message}")
