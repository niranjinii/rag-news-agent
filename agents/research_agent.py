import json
from state import PipelineState, ResearchData
from tools.web_search import ask_llm, generate_subqueries, google_search
from tools.scraper import scrape_and_chunk_recursive
from tools.vector_store import run_hybrid_search

def extract_claim(chunk):
    prompt = f"""
    Extract the single most highly technical, data-driven factual claim from this text into one concise sentence. 
    Focus ONLY on hard metrics, benchmarks, numbers, or architectural specifications. 
    If the text is just marketing fluff, introductory history, or lacks hard technical data, output EXACTLY: 'NO FACTUAL CLAIM'.
    Text: {chunk}
    """
    return ask_llm(prompt).strip().replace('"', '')

def extract_definitions(text_corpus):
    prompt = f"Read this text. Identify up to 2 highly technical terms. Provide a 1-sentence definition for each strictly based on the text. Output ONLY a JSON object where keys are terms and values are definitions. Text: {text_corpus[:3000]}"
    try:
        return json.loads(ask_llm(prompt, response_format="json_object"))
    except:
        return {}

def research_agent_node(state: PipelineState) -> dict:
    """
    Research Agent - pulls live data using modular tools and formats it.
    """
    topic = state["topic"]
    print(f"\n[RESEARCH AGENT] Starting live pipeline for topic: {topic}")
    
    subqueries = generate_subqueries(topic)
    print(f"[RESEARCH AGENT] Generated subqueries: {subqueries}")

    all_chunks = []
    all_metadata = []
    
    for query in subqueries:
        urls = google_search(query)
        for url in urls:
            # THE FIX: Change [:10] to [:50] so it reads past the menus!
            chunks = scrape_and_chunk_recursive(url)[:50] 
            for chunk in chunks:
                if len(chunk) > 100: 
                    all_chunks.append(chunk)
                    all_metadata.append({"url": url, "subtopic": query})
    
    if not all_chunks:
        print("[RESEARCH AGENT] ❌ No data found.")
        return {"research_data": None}

    # Pass the data to your math microservice
    best_matches = run_hybrid_search(subqueries, all_chunks, all_metadata)

    final_sources = []
    source_id_counter = 1
    seen_chunks = set() # <-- CHANGE THIS FROM seen_urls
    
    for match in best_matches:
        raw_chunk = match["chunk"]
        metadata = match["metadata"]
        
        # <-- CHANGE THIS TO CHECK THE CHUNK, NOT THE URL
        if raw_chunk in seen_chunks: 
            continue
            
        claim = extract_claim(raw_chunk)
        if "no factual claim" in claim.lower() or "no fact" in claim.lower():
            continue
            
        final_sources.append({
            "id": source_id_counter,
            "subtopic": metadata["subtopic"],
            "extracted_claim": claim,
            "raw_chunk": raw_chunk,
            "url": metadata["url"]
        })
        
        # <-- TRACK THE CHUNK HERE
        seen_chunks.add(raw_chunk) 
        source_id_counter += 1

    combined_text = " ".join([s["raw_chunk"] for s in final_sources])
    definitions_dict = extract_definitions(combined_text)

    # Package exactly as the new state.py expects
    real_research_data: ResearchData = {
        "definitions": definitions_dict,
        "sources": final_sources
    }

    print(f"[RESEARCH AGENT] ✓ Live research complete! Extracted {len(final_sources)} claims.")
    
    return {
        "research_data": real_research_data
    }