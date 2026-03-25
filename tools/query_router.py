import json
import datetime
from tools.web_search import ask_gemini_gatekeeper

def analyze_and_route_query(topic):
    """
    Acts as the Gatekeeper. Verifies existence (with time awareness) AND determines the research intent.
    """
    current_date = datetime.datetime.now().strftime("%B %Y")
    
    prompt = f"""
    Analyze this user research topic: "{topic}"
    
    CRITICAL TIME CONTEXT: The current date is {current_date}. 
    
    Step 1 (Existence Check): Does this specific product, architecture, or tech concept actually exist and have officially released data as of {current_date}? 
    (For example: Since it is {current_date}, the 'iPhone 16 Pro' and 'M4 MacBook' exist. But the 'PlayStation 7' does NOT exist.)
    
    Step 2 (Intent Classification): Determine the depth of research required based on the topic. Choose ONE:
    - "DEEP_DIVE": Requires searching for complex benchmarks, feature lists, specs, or reviews for a single product.
    - "COMPARISON": Requires searching for head-to-head benchmarks or differences between two or more products (e.g., "M3 vs M2").
    - "QUICK_FACT": A simple, narrow question (e.g., "What is the battery capacity of X?").
    - "REJECT": The product does not exist, is a rumor, or is unreleased as of {current_date}.
    
    Step 3 (Query Generation): If it exists, generate up to 4 highly specific Google search queries tailored to the Intent. If REJECT, output an empty list [].
    
    CRITICAL: Output ONLY a valid JSON object matching this exact schema:
    {{
        "exists": true,
        "existence_reasoning": "Brief explanation of what this product is, and confirmation of its release relative to {current_date}.",
        "intent": "COMPARISON", 
        "search_queries": ["query 1", "query 2"]
    }}
    """
    try:
        raw_response = ask_gemini_gatekeeper(prompt, response_format="json_object")
        return json.loads(raw_response)
    except Exception as e:
        print(f"⚠️ Gatekeeper Error: {e}")
        return {"exists": True, "existence_reasoning": "Failsafe triggered", "intent": "DEEP_DIVE", "search_queries": [topic]}