import streamlit as st
from chatbot_model import get_response

st.title("AI Study Assistant Chatbot")

user_input = st.text_input("Ask a study question:")

if st.button("Ask"):
    if user_input:
        st.write(get_response(user_input))
    else:
        st.warning("Please enter a question")
