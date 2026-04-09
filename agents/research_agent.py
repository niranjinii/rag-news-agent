import json
from pathlib import Path
from state import PipelineState, ResearchData
from tools.web_search import ask_llm, generate_subqueries, google_search, enrich_and_deduplicate
from tools.knowledge_graph import verify_entity_with_wikidata # <-- NEW: Import your API tool
from tools.scraper import scrape_and_chunk, scrape_pdf
from tools.vector_store import run_hybrid_search, rerank_chunks
from tools.query_router import analyze_and_route_query
import time


def _safe_topic_filename(topic: str) -> str:
    """Convert topic text to a filesystem-safe filename stem."""
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in (topic or ""))
    collapsed = "_".join(part for part in normalized.split("_") if part)
    return collapsed or "untitled_topic"


def _log_agent1_output(payload: dict, topic: str) -> None:
    """Persist Agent1 output to research_outputs/<topic>.json."""
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "research_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_topic_filename(topic)}.json"
    try:
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print(f"[RESEARCH AGENT] 📝 Agent1 output saved to: {output_path}")
    except Exception as error:
        print(f"[RESEARCH AGENT] ⚠️ Failed to write Agent1 output JSON: {error}")

def clean_llm_json(raw_str):
    """Bulletproof JSON extractor that ignores conversational filler and markdown."""
    if not raw_str:
        return ""
        
    # Find the first opening bracket and the last closing bracket
    start_idx = raw_str.find('{')
    end_idx = raw_str.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        # Slice out ONLY the valid JSON portion
        clean = raw_str[start_idx:end_idx+1]
        return clean
        
    # Fallback if no brackets are found
    return raw_str.strip()


def extract_claim(chunk: str, original_topic: str, metadata: dict, detected_intent: str, current_guidelines: str) -> str:
    """
    Extracts technical claims using Entity-Binding and Metadata Context.
    """
    # 1. Safely pull the context we need
    source_url = metadata.get("url", "Unknown Source")
    section_headers = metadata.get("Header 1", "") + " " + metadata.get("Header 2", "")
    
    pdf_mode_instruction = ""
    if metadata.get("source_type") == "pdf_whitepaper":
        pdf_mode_instruction = """
        ### 📄 PDF WHITEPAPER MODE ENABLED:
        - This text is from an official technical whitepaper/research paper.
        - Pay EXTREME attention to TABLE DATA. If the text looks like a series of labels and numbers, reconstruct the relationship (e.g., Column Header -> Value).
        - Look for "hidden" specs: layer counts, parameter estimates, training compute (FLOPs), and batch sizes.
        - Treat this as HIGH AUTHORITY. Do not skip dense technical jargon.
        """

    prompt = f"""
    ### ROLE:
    You are a Senior Technical Editor and strict Data Extractor. Your job is to extract 
    HARD, VERIFIABLE FACTS. You despise SEO fluff, marketing "vibes," and generic filler.

    ### INPUTS:
    Target Topic: {original_topic}
    Target Intent: {detected_intent}
    Extraction Goal: {current_guidelines}
    Source URL: {source_url}
    Document Section Context: {section_headers}

    Your goal is to extract the single most important factual claim about the Target Topic based on the Extraction Goal.
    You MUST use the step-by-step Chain-of-Thought reasoning defined in the JSON schema below.

    ### CHAIN-OF-THOUGHT & FILTERING RULES:

    1. SOURCE GRADING & SKEPTICISM: Analyze the text tone and Source URL. Grade authority from 1 to 10.
       - 9-10: Official documentation, spec sheets, whitepapers.
       - 6-8: Established tech journalism, professional reviews.
       - 1-5 (SKEPTICISM MULTIPLIER): User forums, Medium, Zapier. 
         CRITICAL: If a source is in the 1-5 tier, you MUST REJECT anecdotal claims and only accept verifiable specs.

    2. ENTITY ISOLATION: Identify EXACTLY what products are written in the text. Pay attention to tiers (Pro, Max, Base).

    3. DATA-FIRST FILTERING (THE PROXY RULE): Separate Engineering from Subjective Vibes.
       - ACCEPT: Direct architecture specs OR 'Proxy Signals' of underlying system design (e.g., context limits, native multimodality, hardware latency limits, tokenization).
       - REJECT: Subjective adjectives ("feels smoother", "writes better"), high-level marketing fluff, or irrelevant benchmark riddles.
       - If the chunk only contains subjective fluff, mark 'is_relevant': false.

    4. FACT BINDING (Tables): If reading a flattened table, strictly count the column headers and align them to the data values. If the data is misaligned or ambiguous, REJECT the chunk.
       Do not guess which number belongs to which column.

    5. STRICT GROUNDING: Every claim you extract MUST be explicitly written in the provided text. Never guess.

    6. NEGATIVE INTENT (Lack of a Feature): If the Target Topic asks if a feature is LACKING or MISSING:
       - REQUIREMENT: If the text explicitly confirms the feature is removed/missing, extract that fact.
       - THE BAN: Absence of evidence is NOT evidence of absence. Do not infer a negative just because a feature isn't mentioned.
       - THE CORRECTION: If the text explicitly proves the feature ACTUALLY EXISTS (disproving the target topic), you MUST set 'is_relevant': true,
         and extract the claim proving its existence. Prefix the claim with "CORRECTION:" (e.g., "CORRECTION: The product actually includes [Feature X]").

    ### CRITICAL INSTRUCTION: You MUST output ONLY a raw JSON object matching this exact schema. 
    Do NOT write any text, markdown, or preamble before the opening brace. All of your step-by-step reasoning MUST happen INSIDE the JSON string values.

    {{
      "step_1_source_type": "Describe tone, URL type, and apply Skepticism Multiplier if score <= 5.",
      "credibility_score": <int between 1 and 10>,
      "step_2_entities_found": "List the exact product names literally written in the text.",
      "step_3_target_match": "Does the text EXPLICITLY contain the specific feature being asked about? If the user asks if it LACKS a feature, but the text proves it HAS the feature, you MUST write 'CORRECTION TRIGGERED' here.",
      "step_4_data_check": "What is the specific hard metric found? (If none, write 'None').",
      "is_relevant": true, 
      "extracted_claim": "If step_3 says 'CORRECTION TRIGGERED', write: 'CORRECTION: The product actually has [Feature]'. Otherwise, write the factual claim. If no relevant data, set is_relevant to false.",
      "definitions": {{ "jargon_term": "Define a highly technical architecture or proxy term found in the claim." }}
    }}

    Text: {chunk}
    """
    
    # Call the helper function
    raw_response = ask_llm(prompt, response_format="json_object")
    
    cleaned_text = clean_llm_json(raw_response)
    
    try:
        data = json.loads(cleaned_text)
        
        # ==========================================
        # GATE 1: THE CREDIBILITY BOUNCER
        # ==========================================
        credibility = data.get("credibility_score", 0)
        # Ensure it's safely treated as a number
        if isinstance(credibility, str):
            try: credibility = int(credibility)
            except: credibility = 0
            
        if credibility < 6:
            reason = data.get("step_1_source_type", "Unknown")
            print(f"🗑️ Dropped: Source too weak (Score {credibility}/10). AI says: {reason}")
            return "no factual claim" # Kills the chunk!
        
        # ==========================================
        # GATE 2: THE HALLUCINATION & FLUFF BOUNCER
        # ==========================================
        if not data.get("is_relevant", False):
            # I added a print statement here too so you know exactly why it failed relevance!
            match_reason = data.get("step_3_target_match", "Unknown match reason.")
            data_reason = data.get("step_4_data_check", "Unknown data reason.")
            print(f"🗑️ Dropped: Failed relevance check. AI says: {match_reason} | {data_reason}")
            return "no factual claim"
            
        # If it passes BOTH gates, return just the string
        return data.get("extracted_claim", "no factual claim").strip().replace('"', '')
        
    except json.JSONDecodeError as e:
        # Safety net: If the LLM glitches, safely drop the claim without crashing
        print(f"❌ JSON FAIL | Topic: {original_topic[:20]}... | Error: {str(e)}")
        print(f"RAW FAILING TEXT: {raw_response[:150]}...") 
        return "no factual claim"


def research_agent_node(state: PipelineState) -> dict:
    """
    Research Agent - pulls live data using modular tools and formats it.
    Now equipped with a semantic gatekeeper and Knowledge Graph verification!
    """
    topic = state["topic"]
    print(f"\n[RESEARCH AGENT] Starting live pipeline for topic: {topic}")
    
    # ==========================================
    # 1. THE GATEKEEPER
    # ==========================================
    print(f"[RESEARCH AGENT] Analyzing intent and verifying existence...")
    routing_plan = analyze_and_route_query(topic)
    
    # --- NEW: Print the audit trail quote so you can flex the evidence! ---
    extracted_quote = routing_plan.get("exact_quote_from_snippet", "MISSING")
    reasoning = routing_plan.get("existence_reasoning", "MISSING")
    
    print(f"🔒 [GATEKEEPER AUDIT] Evidence: '{extracted_quote}'")
    print(f"🧠 [GATEKEEPER REASONING]: '{reasoning}'")
    # ----------------------------------------------------------------------
    
    # Halt immediately if the product doesn't exist
    if not routing_plan.get("exists", True):
        reasoning = routing_plan.get("existence_reasoning", "No reason provided")
        print(f"[RESEARCH AGENT] 🛑 HALTING RESEARCH: {reasoning}")
        return {"research_data": {"definitions": {}, "sources": []}}
        
    subqueries = routing_plan.get("search_queries", [topic])
    print(f"[RESEARCH AGENT] Intent: {routing_plan.get('intent')} | Domain: {routing_plan.get('subject_domain')}. Using optimized queries: {subqueries}")

    detected_intent = routing_plan.get("intent", "DEEP_DIVE")
    detected_domain = routing_plan.get("subject_domain", "GENERAL_OVERVIEW")

    structure_map = {
        "COMPARISON": "Focus on the 'Delta.' Extract specific differences and reject 'both are good' fluff.",
        "DEEP_DIVE": "Focus on comprehensive technical details for this single product.",
        "QUICK_FACT": "Extract the exact, literal answer to the query.",
        "NEGATIVE_FACT_CHECK": "Focus exclusively on explicit statements of lack, absence, or non-support."
    }

    domain_map = {
        "ARCHITECTURE": "CRITICAL GRACEFUL DEGRADATION: If exact architecture is closed-source, extract 'Proxy Signals' that reflect system design (e.g., native multimodality mechanisms, context window scaling, tokenization limits, or TTFT hardware limits). Reject subjective 'feels smarter' claims.",
        "PRICING": "CRITICAL: Focus on hard financial limits and hidden costs. Reject marketing value statements.",
        "PERFORMANCE_METRICS": "CRITICAL: Extract exact numbers (FPS, Mbps, hours). Reject subjective speed claims.",
        "GENERAL_OVERVIEW": "Extract high-level features and confirmed specs."
    }

    base_rule = structure_map.get(detected_intent, "Extract hard facts.")
    snob_rule = domain_map.get(detected_domain, "")
    current_guidelines = f"{base_rule} {snob_rule}"


    # ==========================================
    # 2. THE SCRAPER & VECTOR PIPELINE
    # ==========================================
    all_chunks = []
    all_metadata = []
    
    seen_urls = set() 
    for query in subqueries:
        urls = google_search(query)
        for url in urls:
            if url in seen_urls:
                continue # Skip duplicates
            seen_urls.add(url)

            # 📍 STEP 1: ROUTE BY FILE TYPE
            # We check if it's a PDF to choose the right scraper
            is_pdf = url.lower().endswith(".pdf") or ".pdf" in url.lower()

            if is_pdf:
                print(f"📄 PDF Detected (Potential Whitepaper): {url}")
                document_chunks = scrape_pdf(url) # Our new PyMuPDF tool
            else:
                # Trafilatura scraper
                document_chunks = scrape_and_chunk(url) 

            # 📍 STEP 2: PROCESS AND MERGE
            for doc in document_chunks:
                text_content = doc.page_content
                
                # Keep only substantial chunks
                if len(text_content) > 100:
                    all_chunks.append(text_content)
                    
                    # 📍 STEP 3: DYNAMIC METADATA
                    # We tag the source so the Extractor knows it's a high-value PDF
                    combined_metadata = {
                        "url": url,
                        "subtopic": query,
                        "source_type": "pdf_whitepaper" if is_pdf else "web_article"
                    }
                    # Header extraction
                    combined_metadata.update(doc.metadata)
                    
                    all_metadata.append(combined_metadata)
    
    if not all_chunks:
        print("[RESEARCH AGENT]  No data found.")
        return {"research_data": {
            "definitions": {},
            "sources": []
        }}

    # 1. Cast the wide net
    best_matches = run_hybrid_search(subqueries, all_chunks, all_metadata)
    
    # THE FILTER: Re-rank the chunks and strictly keep the top 15
    print(f"🎯 Cross-Encoder re-ranking {len(best_matches)} chunks to find the absolute best 15...")
    elite_matches = rerank_chunks(topic, best_matches, top_k=15)

    final_sources = []
    source_id_counter = 1
    seen_chunks = set()
    
    # 2. Only feed the 'elite_matches' to Llama
    for match in elite_matches:
        raw_chunk = match["chunk"]
        metadata = match["metadata"]
        
        # Check if we've seen it, then immediately add it so we never read it twice
        if raw_chunk in seen_chunks:
            continue
        seen_chunks.add(raw_chunk)
        
        # Truncate to ~10k tokens so Groq never chokes on a PDF
        safe_chunk = raw_chunk[:40000] 
                
        claim = extract_claim(
            chunk=safe_chunk, # 📍 UPDATE THIS: Pass safe_chunk instead of raw_chunk
            original_topic=topic, 
            metadata=metadata,
            detected_intent=detected_intent,      
            current_guidelines=current_guidelines 
        )
              
        # Catch nulls, empties, and "no fact" strings all at once
        if not claim or str(claim).strip().lower() == "null" or "no factual claim" in claim.lower() or "no fact" in claim.lower():
            print("🗑️ Dropped: LLM determined the chunk does not contain a valid claim.")
            continue
            
        final_sources.append({
            "id": source_id_counter,
            "subtopic": metadata["subtopic"],
            "extracted_claim": claim,
            "raw_chunk": raw_chunk,
            "url": metadata["url"]
        })
        
        source_id_counter += 1

    # Only run the definition extractor if we actually found facts
    if not final_sources:
        editor_output = {}
    else:
        # ==========================================
        # NEW: THE KNOWLEDGE GRAPH CHECK
        # ==========================================
        # We ping Wikidata using the main topic before passing it to the Editor
        kg_result = verify_entity_with_wikidata(topic)

        # ==========================================
        # THE ENRICHMENT & DEDUPLICATION STEP
        # ==========================================
        print("🧠 Asking Gemini to deduplicate claims and write definitions...")
    
        # NEW: Pass BOTH the extracted sources and the KG result
        editor_output = enrich_and_deduplicate(final_sources, kg_result)

    # ==========================================
    # ASSEMBLE FINAL PAYLOAD
    # ==========================================
    final_sources_list = editor_output.get("sources", editor_output.get("unique_claims", editor_output.get("claims", [])))

    # CLEANUP: Renumber IDs and slice massive chunks to prevent Agent 2 from crashing
    for index, source in enumerate(final_sources_list):
        source["id"] = index + 1
        
        # SLICE: Keep the context but cap the length so the 8b model stays stable.
        # 1,500 chars is enough for evidence without hitting token limits.
        if "raw_chunk" in source and len(source["raw_chunk"]) > 1500:
            source["raw_chunk"] = source["raw_chunk"][:1500] + "... [TRUNCATED FOR TRANSIT]"

    final_output = {
        "research_data": {
            "definitions": editor_output.get("definitions", {}),
            "sources": final_sources_list
        }
    }

    _log_agent1_output(final_output, topic)
    
    print(f"[RESEARCH AGENT] ✓ Live research complete! Crunched down to {len(final_sources_list)} unique facts.")
    return final_output