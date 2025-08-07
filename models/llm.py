from groq import Groq
import streamlit as st
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

def generate_llm_response(chat_history, context, groq_client, response_style="Detailed"):
    query = chat_history[-1]["content"]

    history_str = "\n".join([f'{msg["role"].title()}: {msg["content"]}' for msg in chat_history[:-1]])

    if context:
        context_str = "\n\n".join(context)
        if response_style == "Concise":
            prompt = f"""
            Based *only* on the following context from the student handbook, answer the user's latest question in a single, concise sentence.

            CONTEXT:
            {context_str}

            CONVERSATION HISTORY:
            {history_str}

            LATEST QUESTION:
            {query}

            CONCISE ANSWER:
            """
        else:
            prompt = f"""
            You are an AI assistant. Your primary task is to answer the user's latest question based on the provided context from the student handbook and the conversation history.
            If the question can be answered using the context, form your response based on it.
            If the question is conversational or cannot be answered by the context, answer it using your general knowledge, but give a note like "This answer is from my general knowledge, not the handbook."

            CONTEXT FROM HANDBOOK:
            {context_str}

            CONVERSATION HISTORY:
            {history_str}

            LATEST QUESTION:
            {query}

            DETAILED ANSWER:
            """
    else:
        with st.spinner("Couldn't find an answer in the handbook. Searching the web..."):
            search_results = serpapi_web_search(query)
        
        if not search_results:
            prompt = f"""
            You are an AI assistant. The user asked a question that could not be found in the student handbook or via web search.
            Please answer the following question using your general knowledge and the conversation history. Note that the answer is not from the knowledge base.

            CONVERSATION HISTORY:
            {history_str}
            
            QUESTION:
            {query}

            ANSWER:
            """
        else:
            prompt = f"""
            You are a helpful research assistant. Answer the user's question based on the following web search results and the conversation history. Synthesize the information into a comprehensive answer.

            SEARCH RESULTS:
            {search_results}

            CONVERSATION HISTORY:
            {history_str}

            QUESTION:
            {query}

            ANSWER:
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
