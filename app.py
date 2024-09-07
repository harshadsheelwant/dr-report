import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from transformers import AutoTokenizer
from transformers import pipeline
from streamlit_pdf_viewer import pdf_viewer
import torch
import streamlit_shadcn_ui as ui

# Load the medllama2_7b model and tokenizer from Hugging Face
@st.cache_resource
def load_model():
    model =  "johnsnowlabs/JSL-MedLlama-3-8B-v2.0"
    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer, model

tokenizer, model = load_model()

# Function to generate response using medllama2_7b
def file_preprocessing(file):
    loader =  PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        print(text)
        final_texts = final_texts + text.page_content
    return final_texts

def displayPDF(file):

    with open(file, "rb") as f:
         pdf_viewer(f.read(), height=600, width=800)

def llm(input_text):
  pdf_analysis = pipeline('text-generation',
                      model=model,
                      torch_dtype=torch.float16,
                      device_map="auto",
                      max_length = 5000,
                      min_length = 50)
  analysis = pdf_analysis(input_text)
  analysis = analysis[0]['summary_text']
  return analysis

st.set_page_config(
    page_title="DR-Report",
    page_icon="📄",
    layout="wide",
)

def main():

    st.title("Medical Report Checker")
    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        if ui.button(text="Summarize PDF", key="styled_btn_tailwind_pdf", class_name="bg-orange-500 text-white"):
            col1, col2 = st.columns(2)
            filepath = "data/"+uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded File")
                pdf_view = displayPDF(filepath)
                input_text = file_preprocessing(filepath)
                input_text = input_text[:5000]

            with col2:
                pdf_analysis = llm(input_text)
                st.info("Summarization Complete")
                print(pdf_analysis)
                st.success(pdf_analysis)
    button(username="harshadsheelwant", floating=False, width=221)                
    ui.link_button(text="My LinkedIN", url="https://www.linkedin.com/in/harshadsheelwant/", key="link_btn1", class_name="bg-black hover:bg-blue-500 text-white font-bold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded")
    ui.link_button(text="My Github", url="https://github.com/harshadsheelwant", key="link_btn2", class_name="bg-black shadow-cyan-500/50 hover:bg-blue-500 text-white font-bold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded")


if __name__ == '__main__':
  main()
