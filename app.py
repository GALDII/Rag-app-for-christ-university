import streamlit as st
from config.config import PDF_PATH
from utils.pdf_processor import load_and_chunk_pdf
from models.embeddings import get_clients, setup_vector_store, retrieve_context
from models.llm import get_groq_client, generate_llm_response

def instructions_page():
    st.title("Student Handbook RAG Chatbot")
    st.markdown("Welcome! This is a chatbot designed to answer questions about your student handbook.")
    
    st.markdown("""
    ---
    ## 📝 How to Use
    1.  The student handbook has been pre-loaded into a **Pinecone** vector database.
    2.  Navigate to the **Chat** page using the sidebar.
    3.  Ask any question related to the handbook (e.g., "What is the policy on academic integrity?").
    4.  You can ask follow-up questions like "What are the penalties?" and the chatbot will understand the context.
    5.  If you ask a general question not covered in the handbook, the chatbot will use its general knowledge to answer.
    """)

def chat_page():
    st.title("Christ Handbook Chat")
    st.write("Ask questions about the handbook. You can ask follow-up questions too!")

    @st.cache_resource
    def initialize_rag_pipeline():
        with st.spinner("Connecting to services and setting up the pipeline..."):
            handbook_chunks = load_and_chunk_pdf(PDF_PATH)
            cohere_client, pinecone_client = get_clients()
            vector_store = setup_vector_store(handbook_chunks, cohere_client, pinecone_client)
            groq_client = get_groq_client()
            return cohere_client, vector_store, groq_client

    cohere_client, vector_store, groq_client = initialize_rag_pipeline()

    st.divider()
    response_style_toggle = st.toggle("Get Detailed Answers", value=True)
    response_style = "Detailed" if response_style_toggle else "Concise"


    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating an answer..."):
                context_chunks = retrieve_context(prompt, cohere_client, vector_store)
                answer = generate_llm_response(st.session_state.messages,context_chunks, groq_client, response_style)
                st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        if context_chunks:
            with st.expander("Show sources from the db"):
                for i, text in enumerate(context_chunks):
                    st.info(f"Source {i+1}:\n\n{text}")

        st.rerun()


def main():
    st.set_page_config(
        page_title="Handbook RAG Chat",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"], index=0)
        
        if page == "Chat":
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    if page == "Instructions":
        instructions_page()
    elif page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()