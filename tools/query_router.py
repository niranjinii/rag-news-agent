import json
from tools.web_search import ask_llm

def analyze_and_route_query(topic):
    """
    Acts as the Gatekeeper. Verifies the topic actually exists before wasting API calls.
    """
    prompt = f"""
    Analyze this user research topic: "{topic}"
    
    Step 1 (Existence Check): Does this specific product, architecture, or tech concept actually exist and have officially released data or highly credible leaks? 
    (For example: The 'M3 MacBook Pro' exists. The 'PlayStation 7' does NOT exist.)
    
    Step 2 (Query Generation): If and ONLY if it exists, generate up to 4 highly specific Google search queries to find hard technical benchmarks, spec sheets, and architectural reviews.
    
    CRITICAL: Output ONLY a valid JSON object.
    {{
        "exists": true,
        "existence_reasoning": "Brief explanation of what this product is, or why it is fake/unreleased.",
        "intent": "DEEP_DIVE",
        "search_queries": ["query 1", "query 2", "query 3"]
    }}
    """
    try:
        raw_response = ask_llm(prompt, response_format="json_object")
        return json.loads(raw_response)
    except Exception as e:
        print(f"⚠️ Gatekeeper Error: {e}")
        # Failsafe: if the LLM glitches, let it through just in case it's real
        return {"exists": True, "existence_reasoning": "Failsafe triggered", "search_queries": [topic]}