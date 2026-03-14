import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

def scrape_and_chunk_recursive(url, chunk_size=500, overlap=50):
    """Scrapes URL and splits text intelligently to avoid cutting words/sentences in half."""
    try:
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # Strip UI elements
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        
        # The Pro Upgrade: Recursive Splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""] 
        )
        
        return splitter.split_text(text)
    except:
        return []