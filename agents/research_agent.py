import json
from state import PipelineState, ResearchData
from tools.web_search import ask_llm, generate_subqueries, google_search, enrich_and_deduplicate
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
    You are a strict, literal Data Extractor and Source Judge.
    Target Topic: {original_topic}

    Your goal is to extract the single most important factual claim about the Target Topic, BUT ONLY IF the source is highly credible.
    You MUST use the step-by-step Chain-of-Thought reasoning defined in the JSON schema below to prevent hallucinations and filter out rumors.

    ### CHAIN-OF-THOUGHT & FILTERING RULES:
    1. SOURCE GRADING: Analyze the text tone and source URL (if provided in text). Grade its factual authority from 1 to 10.
       - 9-10: Official documentation, spec sheets, manufacturer announcements.
       - 6-8: Established tech journalism, professional reviews.
       - 1-5: User forums, Reddit, speculative blogs, comment sections, rumors.
    2. ENTITY ISOLATION: Identify EXACTLY what products are explicitly written in the text. Do not infer or guess.
    3. STRICT MATCHING: Does the text explicitly mention the exact Target Topic? (e.g., If the topic is "M4 Max", but the text only says "M4", they DO NOT match).
    4. FACT BINDING: Do NOT "glue" the Target Topic to a stat that belongs to a different product.

    ### CRITICAL INSTRUCTION: You MUST output ONLY valid JSON matching this exact schema. 
    Do not skip steps. The order of these keys acts as your reasoning chain.

    {{
      "step_1_source_type": "Briefly describe the tone and format of this text (e.g., 'Official Specs', 'Tech Review', 'User Forum post').",
      "credibility_score": <int between 1 and 10>,
      "step_2_entities_found": "List the exact product names literally written in the text.",
      "step_3_target_match": "Does the list in Step 2 contain the exact Target Topic? (Yes/No. Explain why.)",
      "step_4_fact_check": "If Yes to Step 3, what is the specific hard metric or feature tied to it?",
      "is_relevant": true, 
      "extracted_claim": "The concise factual claim. Set to 'no factual claim' if is_relevant is false."
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
        
        # ==========================================
        # GATE 1: THE CREDIBILITY BOUNCER
        # ==========================================
        # Safely get the score (in case the LLM outputs a string like "7" instead of an int 7)
        credibility = llm_logic.get("credibility_score", 0)
        if isinstance(credibility, str):
            try: credibility = int(credibility)
            except: credibility = 0
            
        if credibility < 6:
            reason = llm_logic.get("step_1_source_type", "Unknown")
            print(f"🗑️ Dropped: Source too weak (Score {credibility}/10). Llama says: {reason}")
            return "no factual claim" # Kills the chunk!
        
        # ==========================================
        # GATE 2: THE HALLUCINATION BOUNCER
        # ==========================================
        # THE TRAP: If it pulled a competitor's spec, it dies here.
        if not llm_logic.get("is_relevant", False):
            return "no factual claim"
            
        # If it passes BOTH gates, return just the string to keep the rest of your pipeline happy!
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
    
    best_matches = best_matches[:15]

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
        editor_output = {}
    else:
        # ==========================================
        # THE ENRICHMENT & DEDUPLICATION STEP
        # ==========================================
        print("🧠 Asking Gemini to deduplicate claims and write definitions...")
    
        # Pass the list of Llama's extracted sources
        editor_output = enrich_and_deduplicate(final_sources)

    # ==========================================
    # ASSEMBLE FINAL PAYLOAD
    # ==========================================
    # Safely grab the array no matter what it was temporarily called
    final_sources_list = editor_output.get("sources", editor_output.get("unique_claims", editor_output.get("claims", [])))

    final_output = {
        "research_data": {
            "definitions": editor_output.get("definitions", {}),
            "sources": final_sources_list 
        }
    }
    
    print(f"[RESEARCH AGENT] ✓ Live research complete! Crunched down to {len(editor_output.get('unique_claims', []))} unique facts.")
    return final_output