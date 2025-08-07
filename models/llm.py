from groq import Groq
import streamlit as st
from config.config import get_groq_api_key
# Make sure to import the new web search function
from utils.scraper import perform_web_search

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

def generate_llm_response(chat_history, context, groq_client, response_style="Detailed"):
    query = chat_history[-1]["content"]
    history_str = "\n".join([f'{msg["role"].title()}: {msg["content"]}' for msg in chat_history[:-1]])
    

    if not context:
        with st.spinner("Couldn't find an answer in the handbook. Searching christuniversity.in..."):
            context = perform_web_search(query, site_specific=True)

        if not context:
            with st.spinner("No relevant info on the university site. Performing a general web search..."):
                context = perform_web_search(query, site_specific=False)

    if context:
        context_str = "\n\n".join(context) if isinstance(context, list) else context
        prompt = f"""
        You are an AI assistant. Your task is to answer the user's latest question based on the provided context and the conversation history.
        The context could be from a student handbook or from a web page.
        Synthesize the information into a comprehensive answer. If the context is from a webpage, mention the source URL if available.

        CONTEXT:
        {context_str}

        CONVERSATION HISTORY:
        {history_str}

        LATEST QUESTION:
        {query}

        DETAILED ANSWER:
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
            temperature=0.5, # A bit of creativity might be needed for web results
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response from LLM: {e}"
