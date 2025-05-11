import streamlit as st
from transformers import pipeline

# Initialize the Hugging Face model pipeline (GPT-Neo 1.3B, GPT-J 6B, etc.)
# You can choose from a range of models that are small and efficient
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')

# Title of the App
st.title('Q&A Chatbot')

# Sidebar for settings
st.sidebar.title('Settings')

# Domain selection (optional, you can also let users type this)
domain = st.sidebar.text_input('Assistant Expertise Domain (e.g. Python, AI, Math):', value='General Knowledge')

# Main interface for user input
st.write('Go ahead and ask any question')
user_input = st.text_input('You:')

if user_input:
    # Generate the response using the selected Hugging Face model
    try:
        response = generator(user_input, max_length=150, num_return_sequences=1)
        st.write(response[0]['generated_text'])
    except Exception as e:
        st.write(f"Error occurred: {str(e)}")
else:
    st.write('Please provide the query.')

