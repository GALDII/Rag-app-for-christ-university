from groq import Groq
import streamlit as st
from config.config import get_groq_api_key
from utils.web_search import serpapi_web_search

def get_groq_client():
    try:
        groq_api_key = get_groq_api_key()
        if not groq_api_key:
            st.error("Groq API key not found. Please check your configuration.")
            st.stop()
        return Groq(api_key=groq_api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        st.stop()

def generate_llm_response(query, context, groq_client, response_style="Detailed"):
    if context:
        context_str = "\n\n".join(context)
        if response_style == "Concise":
            prompt = f"""
            Based *only* on the following context from the student handbook, answer the user's question in a single, concise sentence.

            CONTEXT:
            {context_str}

            QUESTION:
            {query}

            CONCISE ANSWER:
            """
        else:
            prompt = f"""
            You are a AI assistant. Your primary task is to answer the user's question based on the provided context from the student handbook.
            If the question can be answered using the context, form your response based on it.
            If the question is conversational or cannot be answered by the context, answer it using your general knowledge, but give a note like not from the knowledge base.

            CONTEXT FROM HANDBOOK:
            {context_str}

            QUESTION:
            {query}

            DETAILED ANSWER:
            """
    else:
        with st.spinner("Couldn't find an answer in the handbook. Searching the web..."):
            search_results = serpapi_web_search(query)
        
        if not search_results:
            prompt = f"""
            You are a AI assistant. The user asked a question that could not be found in the student handbook or via web search.
            Please answer the following question using your general knowledge just give note not from knowledge base.

            QUESTION:
            {query}

            ANSWER:
            """
        else:
            if response_style == "Concise":
                prompt = f"""
                Answer the user's question in one concise sentence based on the following web search results.

                SEARCH RESULTS:
                {search_results}

                QUESTION:
                {query}

                CONCISE ANSWER:
                """
            else:
                prompt = f"""
                You are a helpful research assistant. Answer the user's question in detail based on the following web search results. Synthesize the information into a comprehensive answer and cite the links provided.

                SEARCH RESULTS:
                {search_results}

                QUESTION:
                {query}

                DETAILED ANSWER:
                """

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.7 if not context else 0.2,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response from LLM: {e}"
