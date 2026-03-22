import json
import os
from groq import Groq
from dotenv import load_dotenv

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_and_route_query(raw_topic: str) -> dict:
    """
    Acts as a gatekeeper before RAG begins. 
    Classifies the intent and checks for non-existent products.
    """
    
    system_prompt = """
    You are an expert tech-journalism research director. 
    Analyze the user's requested research topic and output a JSON routing plan.
    
    TASK 1: EXISTENCE CHECK
    Does the primary product/entity in this query actually exist as an officially announced or released product? 
    BE EXTREMELY STRICT. If the query mentions a future generation, unreleased model, or rumor 
    (e.g., "Apple M5", "Apple M6", "iPhone 18", "PS6", "M4 Ultra"), you MUST set "exists" to false. Do not assume a product exists just because the company is real.
    
    TASK 2: INTENT CLASSIFICATION
    Classify the query into ONE of these three buckets:
    - SIMPLE_FACT: Asking for a single number, metric, or definition (e.g., "H100 tensor core count", "What is LoRA?").
    - COMPARISON: Pitting two or more specific things against each other (e.g., "M4 Pro vs M4 Max").
    - DEEP_DIVE: A broad request for architecture, features, or general overview (e.g., "DeepSeek R1 capabilities").

    TASK 3: QUERY OPTIMIZATION
    Based on the intent, generate the optimized search queries to feed to the scraper:
    - If SIMPLE_FACT: Return EXACTLY 1 query (just the raw topic).
    - If COMPARISON: Return EXACTLY 1 query that forces a head-to-head search (e.g., "M4 Pro vs M4 Max spec comparison").
    - If DEEP_DIVE: Return 2-3 specific subqueries.

    OUTPUT SCHEMA (Must be valid JSON):
    {
      "existence_reasoning": "Step 1: Explain step-by-step if this product officially exists today, or if it is a future rumor.",
      "exists": true/false,
      "intent": "SIMPLE_FACT" | "COMPARISON" | "DEEP_DIVE",
      "search_queries": ["query1", "query2"]
    }
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", # Use a fast, cheap model for routing
        response_format={"type": "json_object"},
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Topic to analyze: {raw_topic}"}
        ]
    )
    
    return json.loads(response.choices[0].message.content)

# --- QUICK TESTS TO SEE IT IN ACTION ---
if __name__ == "__main__":
    test_topics = [
        "NVIDIA H100 tensor core count",      # Should trigger SIMPLE_FACT
        "Apple M4 Ultra specifications",      # Should trigger exists: false
        "M4 Pro vs M4 Max memory bandwidth"   # Should trigger COMPARISON
    ]

    for topic in test_topics:
        print(f"\nAnalyzing: '{topic}'")
        print(json.dumps(analyze_and_route_query(topic), indent=2))