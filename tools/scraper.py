import trafilatura
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def scrape_and_chunk(url):
    print(f"🕵️‍♂️ Scraping URL: {url}")
    
    # 1. Download and Extract pure Markdown
    downloaded = trafilatura.fetch_url(url)
    
    # include_formatting=True is the magic command that keeps the Markdown headers (##, ###)
    markdown_content = trafilatura.extract(downloaded, include_formatting=True)
    
    if not markdown_content:
        print("⚠️ Warning: Could not extract content from this URL.")
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
    final_chunks = recursive_splitter.split_documents(md_header_splits)
    
    return final_chunks

# --- Quick Test ---
if __name__ == "__main__":
    test_url = "https://en.wikipedia.org/wiki/Large_language_model"
    chunks = scrape_and_chunk(test_url)
    
    print(f"\n✅ Successfully extracted {len(chunks)} smart chunks!")
    
    # Print the first two chunks to see the magic
    for i, chunk in enumerate(chunks[:2]):
        print(f"\n--- CHUNK {i+1} ---")
        print(f"Metadata (Headers): {chunk.metadata}")
        print(f"Content: {chunk.page_content[:500]}...")