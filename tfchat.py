import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="Open Source Chatbot", layout="wide")

# Available Open Source Models
MODEL_OPTIONS = {
    "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.1",
    "Gemma-2B": "google/gemma-1.1-2b-it",
    "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Phi-2": "microsoft/phi-2"
}

@st.cache_resource(show_spinner=True)
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model.eval()
    return tokenizer, model

def generate_response(prompt, tokenizer, model, max_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()

# App title
st.title("🧠 Open Source Chatbot (No API Key)")

# Sidebar settings
st.sidebar.title("⚙️ Settings")
selected_model_name = st.sidebar.selectbox("Select a Model", list(MODEL_OPTIONS.keys()))
max_tokens = st.sidebar.slider("Max Tokens", 50, 1000, 256)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

# Load the selected model
model_id = MODEL_OPTIONS[selected_model_name]
tokenizer, model = load_model(model_id)

# Main Chat UI
st.write("💬 Ask me anything:")
user_input = st.text_input("You:")

if user_input:
    with st.spinner("Generating response..."):
        full_prompt = f"<|user|>: {user_input}\n<|assistant|>:"
        response = generate_response(full_prompt, tokenizer, model, max_tokens, temperature)
        st.write(f"🤖: {response}")
