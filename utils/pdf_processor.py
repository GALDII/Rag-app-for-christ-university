from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

def load_and_chunk_pdf(file_path):
    try:
        pdf_reader = PdfReader(file_path)
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except FileNotFoundError:
        st.error(f"`{file_path}` not found. Please ensure the handbook is in the project root.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during PDF processing: {e}")
        st.stop()

