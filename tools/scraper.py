import requests
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def scrape_and_chunk(url):
    print(f"🕵️‍♂️ Scraping URL via Jina: {url}")
    
    # 1. The Jina AI Cheat Code (Bypasses bot protection & returns Markdown)
    jina_url = f"https://r.jina.ai/{url}"
    
    try:
        # We pass a simple header just to be safe
        response = requests.get(jina_url, headers={'Accept': 'text/event-stream'})
        
        # If Jina hits a completely dead link, it'll throw an error we can catch
        response.raise_for_status() 
        
        markdown_content = response.text
        
        # Jina returns a string. If it's too short, it likely failed.
        if not markdown_content or len(markdown_content) < 50:
            print("⚠️ Warning: Could not extract content from this URL.")
            return []
            
    except Exception as e:
        print(f"⚠️ Warning: Failed to fetch {url}. Error: {e}")
        return []

    # 2. The Semantic Markdown Chunker
    # This tells LangChain to group text logically based on the article's headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # This creates documents where the metadata contains the header it belongs to!
    md_header_splits = markdown_splitter.split_text(markdown_content)

    # 3. The Safety Net (Recursive Splitter)
    # If a specific section under a header is STILL too long for the LLM context window, 
    # we gently slice it down to 1000 characters with a 100 character overlap.
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    
    # Pass the markdown-split documents through the safety net
    # ... (After your recursive_splitter runs) ...
    final_chunks = recursive_splitter.split_documents(md_header_splits)
    
    # ==========================================
    # THE GARBAGE FILTER (NEW)
    # ==========================================
    clean_chunks = []
    for chunk in final_chunks:
        text = chunk.page_content
        
        # 1. Reject if it looks like a JSON blob or CSS (too many curly braces)
        if text.count('{') > 4 and text.count('}') > 4:
            continue
            
        # 2. Reject if it's leaking Next.js or SEO metadata
        if "og:site_name" in text or "{\"title\":" in text or "twitter:card" in text:
            continue
            
        # 3. Reject if it's bizarrely short
        if len(text.strip()) < 30:
            continue
            
        clean_chunks.append(chunk)

    print(f"🧹 Cleaned up {len(final_chunks) - len(clean_chunks)} garbage chunks.")
    return clean_chunks

# --- Quick Test ---
if __name__ == "__main__":
    # Let's test it on a site that usually blocks standard Python requests!
    test_url = "https://www.macrumors.com/roundup/iphone-17-pro/"
    chunks = scrape_and_chunk(test_url)
    
    print(f"\n✅ Successfully extracted {len(chunks)} smart chunks!")
    
    # Print the first two chunks to see the magic
    for i, chunk in enumerate(chunks[:2]):
        print(f"\n--- CHUNK {i+1} ---")
        print(f"Metadata (Headers): {chunk.metadata}")
        print(f"Content: {chunk.page_content[:500]}...")