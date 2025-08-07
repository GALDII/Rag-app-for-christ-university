from groq import Groq
import streamlit as st
from config.config import get_groq_api_key
from utils.scraper import perform_web_search
from models.embeddings import update_vector_store

def get_groq_client():
    """Initializes and returns the Groq client."""
    try:
        groq_api_key = get_groq_api_key()
        if not groq_api_key:
            st.error("Groq API key not found.")
            st.stop()
        return Groq(api_key=groq_api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        st.stop()

def generate_llm_response(chat_history, context, groq_client, cohere_client, index, response_style="Detailed"):
    """
    Generates a direct response as a stream based on a prioritized search strategy.
    """
    query = chat_history[-1]["content"]
    history_str = "\n".join([f'{msg["role"].title()}: {msg["content"]}' for msg in chat_history[:-1]])
    
    source_note = ""
    is_web_search = False

    # If context from the handbook is not found, start the web search process.
    if not context:
        # First, try searching the university website
        site_context = perform_web_search(query, site_specific=True)
        
        # If found on the university site, update the DB and set context
        if site_context:
            update_vector_store(site_context, cohere_client, index)
            context = site_context
            source_note = "Source: christuniversity.in"
            is_web_search = True
        # If not found on the site, then perform a general web search
        else:
            general_web_context = perform_web_search(query, site_specific=False)
            if general_web_context:
                context = general_web_context
                source_note = "Source: Web Search"
                is_web_search = True

    # If context was from the original DB, set the source note.
    elif context:
        source_note = "Source: Student Handbook"

    # Build the final prompt based on whether any context was found.
    if context:
        if isinstance(context, str):
            max_chars = 4000
            context_str = context[:max_chars] if len(context) > max_chars else context
        else:
            context_str = "\n\n".join(context)
        
        prompt = f"""
        **Instructions:**
        1. Answer the "LATEST QUESTION" based *only* on the provided "CONTEXT".
        2. Your answer must be direct and to the point.
        3. Do not add any extra conversational text, apologies, or explanations.
        4. After the answer, add the following "SOURCE NOTE" on a new line.

        **CONTEXT:**
        {context_str}
        **LATEST QUESTION:**
        {query}
        **SOURCE NOTE:**
        {source_note}
        **ANSWER:**
        """
    else:
        # Final fallback: General knowledge
        prompt = f"""
        **Instructions:**
        1. Answer the "LATEST QUESTION" using your general knowledge.
        2. Your answer must be direct and to the point.
        3. After the answer, add "Source: General Knowledge" on a new line.
        
        **LATEST QUESTION:**
        {query}
        **ANSWER:**
        """

    try:
        stream = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.1,
            stream=True,
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""
            
    except Exception as e:
        yield f"Error generating response from LLM: {e}"