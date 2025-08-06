from serpapi import GoogleSearch
from config.config import get_serpapi_api_key
import streamlit as st

def serpapi_web_search(query):
    try:
        serpapi_api_key = get_serpapi_api_key()
        if not serpapi_api_key:
            st.warning("SerpApi API key not found. Web search will be disabled.")
            return ""
            
        params = {
            "q": query,
            "api_key": serpapi_api_key,
            "engine": "google",
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        organic_results = results.get("organic_results", [])
        
        results_str = "\n".join(
            f"Title: {res.get('title', 'N/A')}\nLink: {res.get('link', 'N/A')}\nSnippet: {res.get('snippet', 'N/A')}" 
            for res in organic_results[:3]
        )
        return results_str
        
    except Exception as e:
        st.error(f"An error occurred during web search: {e}")
        return ""
