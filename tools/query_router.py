import json
import datetime
from tools.web_search import ask_gemini_gatekeeper, google_search

def analyze_and_route_query(topic):
    """
    Acts as the Gatekeeper. Verifies existence (with time awareness and live snippets) 
    AND determines the research intent, dynamically rewriting negative queries.
    """
    current_date = datetime.datetime.now().strftime("%B %Y")
    
    # 1. THE PRE-SEARCH: Grab the top 3 Google snippets to reality-check the LLM
    print(f"Running Pre-Search sanity check for: {topic}...")
    try:
        quick_snippets = google_search(f"{topic} official release OR announcement")[:3]
        snippets_text = "\n".join(quick_snippets)
    except:
        snippets_text = "No snippets available."
    
    # 2. THE RUTHLESS PROMPT (Now with Negative Intent routing!)
    prompt = f"""
    You are a strict Gatekeeper and Routing Agent for a tech newsroom.
    Analyze this user research topic: "{topic}"
    
    CRITICAL TIME CONTEXT: The current date is {current_date}. 
    
    Recent Web Snippets for Reality Check:
    {snippets_text}
    
    Step 1 (Existence Check): Does this specific product, architecture, or tech concept actually exist and have officially released data as of {current_date}? 
    - CRITICAL RULE (The Kill-Switch): If the snippets or your internal knowledge explicitly describe the product as "rumored," "leaked," "expected," "unreleased," or "speculated," you MUST mark it as REJECT. Do not hallucinate specs for unreleased products.
    - ESCAPE HATCH RULE (Knowledge Cutoff): Technology moves incredibly fast. If the topic sounds like a highly specific new algorithm, deep-tech architecture, or newly published research paper, and you are simply unsure if it exists due to your training cutoff (and the snippets are inconclusive), you MUST NOT reject it. Output "exists": true and let the live search engine verify it.
    - LEGACY RULE (Version Forgiveness): If the product is a real, older, legacy, or retired version of a technology (e.g., asking about 'Gemini 2.0' when 'Gemini 3.1' exists, or 'Claude 3.5 Opus' when 'Claude 4' exists), DO NOT REJECT IT. Mark as "exists": true. You must allow research on historical benchmarks and superseded models.
    
    Step 2 (Intent Classification): Determine the depth of research required based on the topic. Choose ONE:
    - "DEEP_DIVE": Requires searching for complex benchmarks, feature lists, specs, or reviews for a single product.
    - "COMPARISON": Requires searching for head-to-head benchmarks or differences between two or more products.
    - "QUICK_FACT": A simple, narrow question.
    - "NEGATIVE_FACT_CHECK": The topic asks for what a product LACKS, DOES NOT HAVE, or DOES NOT SUPPORT (e.g., "servers without hyperthreading", "cars lacking lithium batteries").
    - "REJECT": The product does not exist, is a rumor, or is unreleased as of {current_date}.

    Step 3 (Subject Domain Classification): Determine the specific subject matter of the topic. Choose ONE:
    - "ARCHITECTURE": Hardware, internals, system design, thermals, chipsets, neural network specs, or underlying engineering.
    - "PRICING": Costs, subscriptions, tiers, or value propositions.
    - "PERFORMANCE_METRICS": Benchmarks, speed, FPS, battery life testing, throughput, or scores.
    - "GENERAL_OVERVIEW": High-level features, target audience, basic functionality, or aesthetics.
    
    Step 4 (Query Generation): If it exists (NOT REJECT), generate up to 4 highly specific Google search queries tailored to the Intent and Domain. If REJECT, output an empty list [].
    CRITICAL RULE FOR NEGATIVE QUERIES: If the intent is NEGATIVE_FACT_CHECK, DO NOT generate queries like "how to disable" or "troubleshooting". INSTEAD, generate queries to find the core product's "master specifications table", "feature comparison matrix", or "official hardware configuration".
    CRITICAL RULE FOR ARCHITECTURE: If the domain is ARCHITECTURE, ensure at least one query includes terms like "whitepaper", "technical specifications", or "architecture breakdown".
    
    CRITICAL: Output ONLY a valid JSON object matching this exact schema:
    {{
        "exists": true,
        "existence_reasoning": "Brief explanation of what this product is, and confirmation of its release relative to {current_date}, or why it was rejected.",
        "intent": "NEGATIVE_FACT_CHECK", 
        "subject_domain": "ARCHITECTURE",
        "search_queries": ["query 1", "query 2"]
    }}
    """
    try:
        raw_response = ask_gemini_gatekeeper(prompt, response_format="json_object")
        return json.loads(raw_response)
    except Exception as e:
        print(f"⚠️ Gatekeeper Error: {e}")
        return {"exists": True, "existence_reasoning": "Failsafe triggered", "intent": "DEEP_DIVE", "search_queries": [topic]}