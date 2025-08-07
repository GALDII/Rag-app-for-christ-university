import streamlit as st
from config.config import PDF_PATH
from utils.pdf_processor import load_and_chunk_pdf
from models.embeddings import get_clients, setup_vector_store, retrieve_context
from models.llm import get_groq_client, generate_llm_response

def instructions_page():
    """Displays the instructions page."""
    st.title("Student Handbook RAG Chatbot")
    st.markdown("Welcome! This is a chatbot designed to answer questions about your student handbook.")
    
    st.markdown("""
    ---
    ## 📝 How it Works
    1.  Ask any question. The bot will first search its internal knowledge base (the student handbook).
    2.  If the answer isn't found, it will automatically search the official **christuniversity.in** website.
    3.  If new information is found on the website, it's added to the knowledge base to make the bot smarter for future questions.
    4.  If no information is found on the university site, it will perform a general web search as a last resort.
    5.  Each answer includes a note indicating the source of the information.
    """)

def chat_page():
    """Main page for the RAG chatbot interface."""
    st.title("Christ Handbook Chat")
    st.write("Ask questions about the handbook. The bot learns from new information!")

    @st.cache_resource
    def initialize_rag_pipeline():
        """Initializes all necessary clients and data for the RAG pipeline."""
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
            context_chunks = retrieve_context(prompt, cohere_client, vector_store)
            
            response_generator = generate_llm_response(
                chat_history=st.session_state.messages,
                context=context_chunks, 
                groq_client=groq_client, 
                cohere_client=cohere_client,
                index=vector_store,
                response_style=response_style
            )
            full_response = st.write_stream(response_generator)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

def main():
    """Main function to run the Streamlit app with multi-page navigation."""
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
