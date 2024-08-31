import streamlit as st
from transformers import AutoTokenizer, pipeline

@st.cache_resource
def load_pipeline():
    model_name = "ruslanmv/Medical-Llama3-8B"
    
    # Use the Hugging Face pipeline with a small model or hosted inference API
    generator = pipeline("text-generation", model=model_name, tokenizer=model_name, device=0)  # Ensure it's set to CPU if deploying on Streamlit Cloud
    return generator

generator = load_pipeline()

# Function to generate a response using the pipeline
def generate_response(question):
    sys_message = ''' 
    You are an AI Medical Assistant trained on a vast dataset of health information. Please be thorough and
    provide an informative answer. If you don't know the answer to a specific medical inquiry, advise seeking professional help.
    '''   
    input_text = f"{sys_message}\n\nUser: {question}\nAssistant:"
    
    response = generator(input_text, max_new_tokens=100, do_sample=True, pad_token_id=50256)
    answer = response[0]['generated_text'].split("Assistant:")[-1].strip()
    
    return answer

# Streamlit UI
st.title("💬 Medical Chatbot")
st.caption("🚀 Powered by a lightweight model")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Generate response using the pipeline
    msg = generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
