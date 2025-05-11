import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Streamlit page config
st.set_page_config(page_title="Open Chatbot Creator", page_icon="🤖")
st.title("🤖 Build Your Own Chatbot (Free & Offline)")

# Sidebar - chatbot personality and model
st.sidebar.header("🔧 Chatbot Settings")
chatbot_name = st.sidebar.text_input("Bot Name", "ChatterBot")
chatbot_personality = st.sidebar.text_area("Bot Personality", 
    value="You are a helpful, friendly assistant. Answer clearly and politely.")

selected_model = st.sidebar.selectbox("Choose a Model", [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "tiiuae/falcon-7b-instruct",
    "HuggingFaceH4/zephyr-7b-alpha",
    "google/gemma-7b-it",
    "meta-llama/Llama-2-7b-chat-hf"
])

max_tokens = st.sidebar.slider("Max Response Tokens", 100, 1024, 300)
temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7)

# Load model and tokenizer
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

st.info("Loading model... This may take a few seconds the first time.", icon="ℹ️")
generator = load_model(selected_model)

# User input
st.subheader(f"💬 Chat with {chatbot_name}")
user_input = st.text_input("You:", placeholder="Ask a question...")

if user_input:
    full_prompt = f"{chatbot_personality}\nUser: {user_input}\n{chatbot_name}:"
    with st.spinner("Thinking..."):
        response = generator(full_prompt, max_new_tokens=max_tokens, temperature=temperature, do_sample=True)[0]["generated_text"]
        answer = response.split(f"{chatbot_name}:")[-1].strip()
    st.markdown(f"**{chatbot_name}:** {answer}")
