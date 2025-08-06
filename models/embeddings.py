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
            st.error("Cohere API key not found. Please check your configuration.")
            st.stop()
        
        cohere_client = cohere.Client(api_key=cohere_api_key, timeout=60)

        pinecone_api_key = get_pinecone_api_key()
        if not pinecone_api_key:
            st.error("Pinecone API key not found. Please check your .env file.")
            st.stop()
            
        pc = pinecone.Pinecone(api_key=pinecone_api_key)
        return cohere_client, pc
        
    except Exception as e:
        st.error(f"Failed to initialize API clients. Error: {e}")
        st.stop()

def setup_vector_store(chunks, cohere_client, pinecone_client):
    try:
        if PINECONE_INDEX_NAME not in pinecone_client.list_indexes().names():
            st.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
            pinecone_client.create_index(name=PINECONE_INDEX_NAME, dimension=1024, metric='cosine')
            st.success("Index created successfully!")

        index = pinecone_client.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        st.error(f"Failed to get or create Pinecone index: {e}")
        st.stop()

    if index.describe_index_stats()['total_vector_count'] == 0:
        with st.spinner("Embedding handbook... This is a one-time setup."):
            batch_size = 96
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            my_bar = st.progress(0, text="Embedding handbook. Please wait.")

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                current_batch_num = (i // batch_size) + 1
                
                try:
                    response = cohere_client.embed(
                        texts=batch_chunks, model='embed-english-v3.0', input_type='search_document'
                    )
                    embeddings = response.embeddings
                    vectors_to_upsert = [
                        {"id": str(i + j), "values": embedding, "metadata": {"text": chunk}}
                        for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings))
                    ]
                    index.upsert(vectors=vectors_to_upsert)
                    my_bar.progress(min(current_batch_num / total_batches, 1.0), text=f"Processed batch {current_batch_num}/{total_batches}")
                except Exception as e:
                    st.error(f"An error occurred during embedding batch {current_batch_num}: {e}")
                    my_bar.empty()
                    st.stop()
            
            my_bar.empty()
            st.success("Handbook embedded successfully!")
            
    return index

def retrieve_context(query, cohere_client, index, n_results=5):
    try:
        response = cohere_client.embed(
            texts=[query], model='embed-english-v3.0', input_type='search_query'
        )
        query_embedding = response.embeddings[0]
        
        results = index.query(
            vector=query_embedding, top_k=n_results, include_metadata=True
        )
        
        similarity_threshold = 0.3
        
        filtered_matches = [match for match in results['matches'] if match['score'] > similarity_threshold]
        
        if not filtered_matches:
            return []
            
        context = [match['metadata']['text'] for match in filtered_matches]
        return context
        
    except Exception as e:
        st.error(f"Failed to retrieve context from the vector store: {e}")
        return []
