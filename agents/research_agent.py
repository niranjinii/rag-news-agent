import json
from state import PipelineState, ResearchData
from tools.web_search import ask_llm, generate_subqueries, google_search
from tools.scraper import scrape_and_chunk
from tools.vector_store import run_hybrid_search
from tools.query_router import analyze_and_route_query
import time

def extract_claim(chunk: str, original_topic: str) -> str:
    """
    Extracts technical claims but forces Entity-Binding to prevent hallucinating specs 
    from the wrong products. Returns a standard string for the pipeline.
    """
    prompt = f"""
    Analyze this text chunk based on the user's research topic: "{original_topic}"
    
    You must extract the single most highly technical, data-driven factual claim from this text into one concise sentence. 
    Focus ONLY on hard metrics, benchmarks, numbers, or architectural specifications.
    
    CRITICAL ENTITY BINDING: The metric you extract MUST belong exactly to the product requested in the topic ("{original_topic}"). If the text highlights an impressive metric for a predecessor (like M1/M2) or a competitor, DO NOT extract it. 
    
    CRITICAL INSTRUCTION: You MUST output your answer in EXACT, valid JSON.
    
    {{
      "relevance_reasoning": "Step 1: Does this text contain a hard metric specifically for {original_topic}? Step 2: Ensure the metric actually belongs to {original_topic} and not a comparison product.",
      "is_relevant": true,
      "extracted_claim": "The raw technical fact. Set to 'no factual claim' if is_relevant is false, or if the only hard data belongs to a different product."
    }}
    
    Text: {chunk}
    """
    
    # Call your existing helper function
    raw_response = ask_llm(prompt).strip()
    
    try:
        # Clean up in case the LLM wraps the JSON in markdown blocks (e.g., ```json ... ```)
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3].strip()
        elif raw_response.startswith("```"):
            raw_response = raw_response[3:-3].strip()
            
        # Parse the LLM's "thought process"
        llm_logic = json.loads(raw_response)
        
        # THE TRAP: If it pulled a competitor's spec, it dies here.
        if not llm_logic.get("is_relevant", False):
            return "no factual claim"
            
        # If it passes, return just the string to keep the rest of your pipeline happy!
        return llm_logic.get("extracted_claim", "no factual claim").strip().replace('"', '')
        
    except json.JSONDecodeError:
        # Safety net: If the LLM glitches and doesn't write JSON, drop the claim
        print(f"[EXTRACTOR WARNING] LLM failed to return valid JSON. Dropping chunk.")
        return "no factual claim"

def extract_definitions(text_corpus):
    prompt = f"Read this text. Identify up to 2 highly technical terms. Provide a 1-sentence definition for each strictly based on the text. Output ONLY a JSON object where keys are terms and values are definitions. Text: {text_corpus[:3000]}"
    try:
        return json.loads(ask_llm(prompt, response_format="json_object"))
    except:
        return {}

def research_agent_node(state: PipelineState) -> dict:
    """
    Research Agent - pulls live data using modular tools and formats it.
    Now equipped with a semantic gatekeeper to prevent hallucinations!
    """
    topic = state["topic"]
    print(f"\n[RESEARCH AGENT] Starting live pipeline for topic: {topic}")
    
    # ==========================================
    # 1. THE GATEKEEPER (New Routing Logic)
    # ==========================================
    print(f"[RESEARCH AGENT] Analyzing intent and verifying existence...")
    routing_plan = analyze_and_route_query(topic)
    
    # Halt immediately if the product doesn't exist (Fixes FM-3)
    # Halt immediately if the product doesn't exist
    if not routing_plan.get("exists", True):
        # Changed 'reason' to 'existence_reasoning'
        reasoning = routing_plan.get("existence_reasoning", "No reason provided")
        print(f"[RESEARCH AGENT] 🛑 HALTING RESEARCH: {reasoning}")
        return {
            "research_data": {
                "definitions": {}, 
                "sources": []
            }
        }
        
    # Use the perfectly optimized queries (Fixes FM-5)
    subqueries = routing_plan.get("search_queries", [topic])
    print(f"[RESEARCH AGENT] Intent: {routing_plan.get('intent')}. Using optimized queries: {subqueries}")

    # ==========================================
    # 2. THE SCRAPER & VECTOR PIPELINE
    # ==========================================
    all_chunks = []
    all_metadata = []
    
    seen_urls = set() # Add this right above the loop!
    for query in subqueries:
        urls = google_search(query)
        for url in urls:
            if url in seen_urls:
                continue # Skip if we already scraped it!
            seen_urls.add(url)
            # Get the LangChain Document objects from our smart scraper
            document_chunks = scrape_and_chunk(url)
            
            for doc in document_chunks:
                # 1. Grab the actual text string
                text_content = doc.page_content
                
                # 2. Check the length of the string, not the object!
                if len(text_content) > 100: 
                    all_chunks.append(text_content)
                    
                    # 3. Merge our URL info with the Markdown Headers!
                    combined_metadata = {
                        "url": url, 
                        "subtopic": query
                    }
                    # This adds things like {'Header 2': 'Battery Specs'} so the vector store has it
                    combined_metadata.update(doc.metadata) 
                    
                    all_metadata.append(combined_metadata)
    
    if not all_chunks:
        print("[RESEARCH AGENT]  No data found.")
        return {"research_data": {
            "definitions": {}, 
            "sources": []
        }}

    # Pass the data to your math microservice
    best_matches = run_hybrid_search(subqueries, all_chunks, all_metadata)

    final_sources = []
    source_id_counter = 1
    seen_chunks = set() 
    
    for match in best_matches:
        raw_chunk = match["chunk"]
        metadata = match["metadata"]
            
        if raw_chunk in seen_chunks: 
            continue
                
        # The Heavy LLM Call
        claim = extract_claim(raw_chunk, topic)
            
        # THE FIX: Add the speed bump immediately after!
        print("[EXTRACTOR] Sleeping for 4 seconds to respect API limits...")
        time.sleep(4) 
            
        if "no factual claim" in claim.lower() or "no fact" in claim.lower():
            continue
            
        final_sources.append({
            "id": source_id_counter,
            "subtopic": metadata["subtopic"],
            "extracted_claim": claim,
            "raw_chunk": raw_chunk,
            "url": metadata["url"]
        })
        
        # Track the chunk to prevent duplicates
        seen_chunks.add(raw_chunk) 
        source_id_counter += 1

    # Only run the definition extractor if we actually found facts!
    if not final_sources:
        definitions_dict = {}
    else:
        combined_text = " ".join([s["raw_chunk"] for s in final_sources])
        definitions_dict = extract_definitions(combined_text)

    # Package exactly as the state.py expects
    real_research_data: ResearchData = {
        "definitions": definitions_dict,
        "sources": final_sources
    }

    print(f"[RESEARCH AGENT] ✓ Live research complete! Extracted {len(final_sources)} claims.")
    
    return {
        "research_data": real_research_data
    }