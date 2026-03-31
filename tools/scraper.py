import requests
import io
import pymupdf
import trafilatura
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def scrape_pdf(url):
    """Downloads a PDF and extracts text using the modern PyMuPDF namespace."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        # Open the PDF from memory using the new namespace
        with pymupdf.open(stream=io.BytesIO(response.content), filetype="pdf") as doc:
            # We join pages with a form feed character
            full_text = chr(12).join([page.get_text() for page in doc])
            
            # Returning as a LangChain Document so it slides right into your loop
            return [Document(
                page_content=full_text, 
                metadata={
                    "url": url, 
                    "source_type": "pdf_whitepaper",
                    "page_count": len(doc)
                }
            )]
            
    except Exception as e:
        print(f"❌ PDF Scraping Error on {url}: {e}")
        return []

def scrape_and_chunk(url):
    print(f"🕵️‍♂️ Fetching via Jina, Extracting via Trafilatura: {url}")
    
    jina_url = f"https://r.jina.ai/{url}"
    
    try:
        # 1. THE MUSCLE: Ask Jina to bypass bots and return RAW HTML
        headers = {
            'X-Return-Format': 'html',
            'Accept': 'text/html'
        }
        response = requests.get(jina_url, headers=headers)
        response.raise_for_status() 
        raw_html = response.text
        
        # 2. THE SCALPEL: Trafilatura cleans the HTML and outputs pure Markdown!
        # output_format="markdown" is the modern Trafilatura way to do this
        markdown_content = trafilatura.extract(raw_html, output_format="markdown")
        
        if not markdown_content or len(markdown_content) < 50:
            print("⚠️ Warning: Trafilatura could not find a clean article here.")
            return []
            
    except Exception as e:
        print(f"⚠️ Warning: Failed to fetch {url}. Error: {e}")
        return []

    # 3. THE SMART CHUNKER: Group logically by headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_content)

    # 4. THE SAFETY NET: Slice anything still too big
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    
    final_chunks = recursive_splitter.split_documents(md_header_splits)
    
    # Inject the URL into the metadata
    for chunk in final_chunks:
        chunk.metadata["url"] = url
        
    return final_chunks