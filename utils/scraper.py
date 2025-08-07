import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
import os
import streamlit as st

def perform_web_search(query: str, site_specific: bool = False, max_pages: int = 2) -> str:
    """Performs a web search and scrapes top results without displaying status messages."""
    if site_specific:
        search_query = f"site:christuniversity.in {query}"
    else:
        search_query = query

    try:
        params = {
          "api_key": os.environ.get("SERPAPI_API_KEY"),
          "engine": "google",
          "q": search_query,
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        
        if not organic_results:
            return ""

        scraped_texts = []
        urls_to_scrape = [result['link'] for result in organic_results[:max_pages]]
        
        for url in urls_to_scrape:
            try:
                response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    element.decompose()
                text = soup.get_text(separator=' ', strip=True)
                scraped_texts.append(f"--- Content from {url} ---\n{text}")
            except requests.RequestException:
                continue

        return "\n\n".join(scraped_texts)

    except Exception:
        return ""