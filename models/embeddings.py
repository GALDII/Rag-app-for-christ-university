import cohere
import pinecone
import streamlit as st
import time
import ssl
from config.config import (
    get_cohere_api_key, 
    get_pinecone_api_key,
    PINECONE_INDEX_NAME
)

def get_clients():
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        cohere_api_key = get_cohere_api_key()
        if not cohere_api_key:
            st.error("Cohere API key not found.")
            st.stop()
        
        cohere_client = cohere.Client(api_key=cohere_api_key, timeout=60)

        pinecone_api_key = get_pinecone_api_key()
        if not pinecone_api_key:
            st.error("Pinecone API key not found.")
            st.stop()
            
        pc = pinecone.Pinecone(api_key=pinecone_api_key)
        return cohere_client, pc
        
    except Exception as e:
        st.error(f"Failed to initialize API clients. Error: {e}")
        st.stop()

def setup_vector_store(chunks, cohere_client, pinecone_client):
    try:
        if PINECONE_INDEX_NAME not in pinecone_client.list_indexes().names():
            pinecone_client.create_index(name=PINECONE_INDEX_NAME, dimension=1024, metric='cosine')

        index = pinecone_client.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        st.error(f"Failed to get or create Pinecone index: {e}")
        st.stop()

    if index.describe_index_stats()['total_vector_count'] == 0:
        with st.spinner("Embedding handbook... This is a one-time setup."):
            batch_size = 96
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                try:
                    response = cohere_client.embed(texts=batch_chunks, model='embed-english-v3.0', input_type='search_document')
                    embeddings = response.embeddings
                    vectors_to_upsert = [{"id": str(i + j), "values": embedding, "metadata": {"text": chunk}} for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings))]
                    index.upsert(vectors=vectors_to_upsert)
                except Exception as e:
                    st.error(f"An error occurred during embedding: {e}")
                    st.stop()
    return index

def retrieve_context(query, cohere_client, index, n_results=5):
    try:
        response = cohere_client.embed(texts=[query], model='embed-english-v3.0', input_type='search_query')
        query_embedding = response.embeddings[0]
        results = index.query(vector=query_embedding, top_k=n_results, include_metadata=True)
        
        similarity_threshold = 0.6
        
        filtered_matches = [match for match in results['matches'] if match['score'] > similarity_threshold]
        if not filtered_matches:
            return []
        context = [match['metadata']['text'] for match in filtered_matches]
        return context
    except Exception as e:
        st.error(f"Failed to retrieve context: {e}")
        return []

def update_vector_store(new_text: str, cohere_client, index):
    """Chunks new text, embeds it, and upserts it into the Pinecone index."""
    try:
        st.info("New information found. Updating knowledge base...")
        
        chunks = [p.strip() for p in new_text.split('\n\n') if p.strip() and len(p) > 50]
        if not chunks:
            return

        current_stats = index.describe_index_stats()
        base_id = current_stats.get('total_vector_count', 0)

        batch_size = 96
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            response = cohere_client.embed(
                texts=batch_chunks, model='embed-english-v3.0', input_type='search_document'
            )
            embeddings = response.embeddings
            vectors_to_upsert = [
                {"id": f"web_{base_id + i + j}", "values": embedding, "metadata": {"text": chunk}}
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings))
            ]
            index.upsert(vectors=vectors_to_upsert)
        
        st.success("Knowledge base updated successfully!")

    except Exception as e:
        st.error(f"Failed to update vector store: {e}")
