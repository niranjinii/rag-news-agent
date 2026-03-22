import os
import json
from google import genai
from google.genai import types

def analyze_and_route_query(raw_topic: str) -> dict:
    """
    Acts as a live-grounded gatekeeper using Gemini 2.5 Flash.
    Checks the actual internet to see if products exist before routing.
    """
    
    system_prompt = """
    You are an expert tech-journalism research director. 
    Analyze the user's requested research topic and output a JSON routing plan.
    
    TASK 1: EXISTENCE CHECK
    Does the primary product/entity in this query actually exist as an officially announced or released product? 
    If the query mentions a purely fictional rumor or joke (like "iPhone 100" or "M9 Ultra"), set "exists" to false.
    If it's a real product or a highly plausible upcoming release (like "iPhone 17"), set "exists" to true.

    TASK 2: INTENT CLASSIFICATION
    Classify the query into ONE of these three buckets:
    - SIMPLE_FACT: Asking for a single number, metric, or definition.
    - COMPARISON: Pitting two or more specific things against each other.
    - DEEP_DIVE: A broad request for architecture, features, or general overview.

    CRITICAL INSTRUCTION: You MUST output your final answer as EXACT, valid JSON. Do not include any introductory or concluding text.
    {
      "existence_reasoning": "Explain if this product officially exists or is an upcoming release based on live search.",
      "exists": true,
      "intent": "DEEP_DIVE",
      "search_queries": ["query1", "query2"]
    }
    """

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    print(f"[GATEKEEPER] Consulting Gemini 2.5 Flash for: '{raw_topic}'...")
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Topic to analyze: {raw_topic}",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
                # Notice we removed the response_schema and mime_type!
                tools=[{"google_search": {}}] 
            )
        )
        
        # Clean up the response in case Gemini wraps it in ```json ... ``` markdown
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()
            
        return json.loads(raw_text)
        
    except Exception as e:
        print(f"[GATEKEEPER ERROR] {e}")
        # Failsafe: Let it pass to the scraper if it crashes
        return {
            "exists": True, 
            "existence_reasoning": "API Fallback", 
            "intent": "DEEP_DIVE", 
            "search_queries": [raw_topic]
        }

# --- QUICK TEST ---
if __name__ == "__main__":
    print(json.dumps(analyze_and_route_query("Apple iPhone 17"), indent=2))