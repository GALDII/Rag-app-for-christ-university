from groq import Groq
import streamlit as st
from config.config import get_groq_api_key
from utils.scraper import perform_web_search

def get_groq_client():
    """Initializes and returns the Groq client, handling potential errors."""
    try:
        groq_api_key = get_groq_api_key()
        if not groq_api_key:
            st.error("Groq API key not found. Please check your configuration.")
            st.stop()
        return Groq(api_key=groq_api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        st.stop()

def generate_llm_response(chat_history, context, groq_client, response_style="Detailed"):
    """
    Generates a response from the LLM based on chat history and a multi-layered context retrieval strategy.
    """
    query = chat_history[-1]["content"]
    history_str = "\n".join([f'{msg["role"].title()}: {msg["content"]}' for msg in chat_history[:-1]])
    
    if not context:
        with st.spinner("Couldn't find an answer in the handbook. Searching christuniversity.in..."):
            context = perform_web_search(query, site_specific=True)

        if not context:
            with st.spinner("No relevant info on the university site. Performing a general web search..."):
                context = perform_web_search(query, site_specific=False)

    if context:
        if isinstance(context, str):
            max_chars = 4000
            if len(context) > max_chars:
                st.warning(f"Web content was too long, truncating to {max_chars} characters for the LLM.")
                context_str = context[:max_chars]
            else:
                context_str = context
        else:
            context_str = "\n\n".join(context)
        
        prompt = f"""
        You are a highly skilled information extraction assistant. Your ONLY task is to find the direct and specific answer to the "LATEST QUESTION" using the provided "CONTEXT".

        - Read the "LATEST QUESTION" carefully to understand what specific piece of information is being asked for.
        - Scrutinize the "CONTEXT" to find the exact answer.
        - If you find the answer, provide it directly and concisely.
        - If the "CONTEXT" does not contain the answer, state clearly that you could not find the specific information in the provided text. Do not suggest other ways to find the information.

        CONTEXT:
        {context_str}

        CONVERSATION HISTORY:
        {history_str}

        LATEST QUESTION:
        {query}

        PRECISE ANSWER:
        """
    else:
        prompt = f"""
        You are an AI assistant. The user asked a question that could not be answered from the student handbook or any web search.
        Please answer the following question using your general knowledge, and clearly state that the information is not from an official source.

        CONVERSATION HISTORY:
        {history_str}
        
        QUESTION:
        {query}

        ANSWER (from general knowledge):
        """

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.2, # Lower temperature for more factual extraction
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response from LLM: {e}"