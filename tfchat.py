import streamlit as st
from transformers import pipeline

# Load the lightweight model (free + no API key)
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="google/flan-t5-small")

generator = load_model()

# Title
st.title("Simple Chatbot (Offline, No API)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
user_input = st.chat_input("Ask something...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        bot_response = generator(user_input, max_length=100, num_return_sequences=1)[0]['generated_text']
        # Post-process to remove user input from output
        bot_reply = bot_response[len(user_input):].strip()
        st.markdown(bot_reply)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
