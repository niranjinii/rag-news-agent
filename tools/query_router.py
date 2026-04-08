import json
import datetime
from tools.web_search import ask_gemini_gatekeeper, google_search

def analyze_and_route_query(topic):
    """
    Acts as the Gatekeeper. Verifies existence using evidence-based extraction 
    to bypass LLM memory conflicts, AND determines the research intent.
    """
    current_date = datetime.datetime.now().strftime("%B %Y")
    
    # 1. THE PRE-SEARCH: Grab the top 3 Google snippets to reality-check the LLM
    print(f"Running Pre-Search sanity check for: {topic}...")
    try:
        quick_snippets = google_search(f"{topic} official release OR announcement")[:3]
        snippets_text = "\n".join(quick_snippets)
    except:
        snippets_text = "No snippets available."
    
    # 2. THE PROMPT
    prompt = f"""
    You are a strict Gatekeeper and Routing Agent for a tech newsroom.
    Analyze this user research topic: "{topic}"
    
    CRITICAL TIME CONTEXT: The current date is {current_date}. 
    
    Recent Web Snippets for Reality Check:
    {snippets_text}
    
    Step 1 (Existence Check): Does this specific product, architecture, or tech concept actually exist and have officially released data? 
    - CRITICAL RULE (Ignore Internal Memory): You MUST base your release status decision STRICTLY on the 'Recent Web Snippets' provided above. Do not use your internal training data to second-guess the live snippets. If the snippets say it exists, it exists.
    - EXTRACT EVIDENCE: You must extract the exact quote from the snippets that justifies your decision to approve or reject. 
    - EVIDENCE HIERARCHY & DATE AWARENESS: Distinguish between "High-Weight" evidence (Retail listings, 'Full Reviews', 'Official Specs', 'Launch Coverage' from 2026) and "Low-Weight" evidence ('Leaks', 'Rumors', 'Upcoming' mentions from 2025). 
    - THE OVERRIDE RULE: If snippets are mixed (e.g., an old 2025 snippet says "Upcoming" but a newer 2026 snippet shows a "Review" or "Buy Now" link), the High-Weight, newer evidence MUST override the older labels. 
    - THE KILL-SWITCH: Mark as REJECT ("exists": false) ONLY if the snippets are dominated by "rumored," "leaked," "expected," "unreleased," or "speculated" tags AND you cannot find a single piece of High-Weight evidence confirming it is currently available as of {current_date}.
    - ESCAPE HATCH RULE (Knowledge Cutoff): ONLY apply this if the topic is a highly specific software algorithm, research paper, or deep-tech architecture. If the topic is a mainstream consumer hardware product (e.g., phones, consoles, GPUs) and the snippets do NOT explicitly confirm an official release, you MUST assume it is unreleased and REJECT IT ("exists": false).
    - LEGACY RULE (Version Forgiveness): If the product is a real, older, legacy, or retired version of a technology, DO NOT REJECT IT. Mark as "exists": true. You must allow research on historical benchmarks.
    
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
        "exact_quote_from_snippet": "The verbatim text from the snippets proving it is released or a rumor. Output 'None' if snippets are empty.",
        "existence_reasoning": "Analyze the quote. Does it indicate a released product or a future/rumored one?",
        "exists": true, 
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
        return {"exists": True, "exact_quote_from_snippet": "None", "existence_reasoning": "Failsafe triggered", "intent": "DEEP_DIVE", "subject_domain": "GENERAL_OVERVIEW", "search_queries": [topic]}