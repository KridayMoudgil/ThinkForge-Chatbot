from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import streamlit as st

# Set up Streamlit app title
st.title("Create and chat with your own chatbot.")

# Load pre-trained Tiny Llama model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained("TinyL1ama/TinyL1ama-1.1B -Chat-vl.")
model = LlamaForCausalLM.from_pretrained("TinyL1ama/TinyL1ama-1.1B -Chat-vl.")

# Initialize context and conversation history
context = st.text_input("What is the purpose you want your chatbot for?")
conversation_history = context + "\n"  # Start conversation with context

# Input field for user query
user_input = st.text_input("You: ")

if user_input:
    # Append user input to the conversation history
    conversation_history += f"You: {user_input}\n"

    # Encode the conversation history and user input
    inputs = tokenizer(conversation_history, return_tensors="pt", truncation=True, max_length=1024)

    # Generate a response from the model
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=1024, num_return_sequences=1)

    # Decode the model's response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the response part from the model's output
    model_response = response[len(conversation_history):].strip()

    # Display model's response and update conversation history
    st.write(f"Tiny Llama: {model_response}")
    conversation_history += f"Tiny Llama: {model_response}\n"
