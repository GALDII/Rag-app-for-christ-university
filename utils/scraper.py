import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
import os
import streamlit as st

def perform_web_search(query: str, site_specific: bool = False, max_pages: int = 2) -> str:
    """
    Performs a web search, scrapes the top results, and returns the combined text content.
    """
    search_domain = "christuniversity.in"
    if site_specific:
        search_query = f"site:{search_domain} {query}"
        st.write(f"Searching for '{query}' on {search_domain}...")
    else:
        search_query = query
        st.write(f"Performing a general web search for '{query}'...")

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
            st.warning(f"No relevant pages found for '{query}' during this search.")
            return ""

        scraped_texts = []
        urls_to_scrape = [result['link'] for result in organic_results[:max_pages]]
        
        st.write(f"Found {len(urls_to_scrape)} relevant pages. Reading content...")

        for url in urls_to_scrape:
            try:
                response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    element.decompose()
                
                text = soup.get_text(separator=' ', strip=True)
                scraped_texts.append(f"--- Content from {url} ---\n{text}")

            except requests.RequestException as e:
                st.error(f"Error scraping {url}: {e}")
                continue

        return "\n\n".join(scraped_texts)

    except Exception as e:
        st.error(f"An error occurred during web search: {e}")
        return ""
