import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the medllama2_7b model and tokenizer from Hugging Face
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("llSourcell/medllama2_7b")
    model = AutoModelForCausalLM.from_pretrained("llSourcell/medllama2_7b")
    return tokenizer, model

tokenizer, model = load_model()

# Function to generate response using medllama2_7b
def generate_response(messages):
    # Prepare the conversation as input for the model
    conversation = ""
    for msg in messages:
        conversation += f"{msg['role']}: {msg['content']}\n"
    
    inputs = tokenizer(conversation, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=500, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs["input_ids"].shape[-1]:][0], skip_special_tokens=True)
    
    return response.strip()


st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by medllama2_7b")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Generate response using medllama2_7b model
    msg = generate_response(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

