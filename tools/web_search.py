import os
import json
import requests
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types
from groq import Groq

# 1. Force Python to hunt down the .env file wherever it is
env_path = find_dotenv()
print(f"DEBUG: Found .env file at: {env_path}")
load_dotenv(env_path)

# 2. Grab the keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# 3. Safety checks
if not GEMINI_API_KEY:
    raise ValueError("🚨 STOP: GEMINI_API_KEY is missing! Check your .env file.")
if not GROQ_API_KEY:
    raise ValueError("🚨 STOP: GROQ_API_KEY is missing! Check your .env file.")
if not SERPER_API_KEY:
    raise ValueError("🚨 STOP: SERPER_API_KEY is missing! Check your .env file.")

# 4. Initialize both clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)


# --- THE WORKER (GROQ / LLAMA 3.3 70B) ---
def ask_llm(prompt, response_format="text"):
    """Global wrapper for Groq LLM calls. Used for heavy chunk extraction."""
    try:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": "llama-3.1-8b-instant",
            "messages": messages,
            "temperature": 0.0 # Low temp for strict facts!
        }
        
        # Force strict JSON mode if requested
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
            
        response = groq_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ Groq API Error: {e}")
        return "{}" if response_format == "json_object" else ""


# --- THE MANAGER (GEMINI) ---
def ask_gemini_gatekeeper(prompt, response_format="text"):
    """Used ONLY for the Query Router to check if products exist."""
    try:
        config = types.GenerateContentConfig(temperature=0.1)
        if response_format == "json_object":
            config.response_mime_type = "application/json"

        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt,
            config=config
        )
        return response.text
    except Exception as e:
        print(f"⚠️ Gemini API Error: {e}")
        return "{}" if response_format == "json_object" else ""


def generate_subqueries(topic):
    """Dynamically categorizes the topic to generate relevant technical queries."""
    prompt = f"""
    Topic: {topic}
    
    You are a Senior Technical Researcher. First, determine the domain of this topic 
    (e.g., AI/Software, Hardware/Silicon, Biotech, Energy, or General Engineering).
    
    Then, generate 3 highly specific Google search queries that target:
    1. Technical Architecture or Mechanism (how it works)
    2. Quantitative Benchmarks or Performance Metrics (real-world data)
    3. Comparison with previous generation or industry standards.

    Output ONLY a JSON object: {{"queries": ["q1", "q2", "q3"]}}
    """
    try:
        # Uses Llama 70B by default!
        response = ask_llm(prompt, response_format="json_object")
        return json.loads(response).get("queries", [topic])
    except Exception as e:
        print(f"⚠️ Error parsing subqueries: {e}")
        return [topic]


def google_search(query):
    """Fetches valid URLs using Serper API, avoiding garbage domains."""
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": 5}) 
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    blacklist = ["reddit.com", "quora.com", "youtube.com", "facebook.com", "twitter.com", "x.com"]
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        results = response.json()
        valid_urls = []
        for item in results.get('organic', []):
            if 'link' in item and not any(bad in item['link'] for bad in blacklist):
                valid_urls.append(item['link'])
                if len(valid_urls) == 5: 
                    break
        return valid_urls
    except Exception as e:
        print(f"⚠️ Serper API error: {e}")
        return []
    
def enrich_and_deduplicate(claims_data):
    """
    The Final Editor Step: Uses Gemini as a decision engine to identify unique IDs, 
    but relies on Python to stitch the original metadata back together.
    """
    if not claims_data:
        return {"sources": [], "definitions": {}}

    # 1. Strip the heavy payload! Only send the ID and the claim to Gemini.
    lite_claims = [{"id": item["id"], "claim": item["extracted_claim"]} for item in claims_data]
    claims_json_str = json.dumps(lite_claims, indent=2)
    
    prompt = f"""
    You are a Senior Technical Editor. Review this list of factual claims:
    
    {claims_json_str}
    
    ### Task 1: Deduplication
    Identify which claims are completely unique. If there are duplicates or claims saying the exact same metric, pick the best-worded one and discard the rest.
    Return a list of the EXACT integer "id"s of the claims that should survive.
    
    ### Task 2: Dictionary
    Identify 2-3 complex technical terms or jargon from the surviving claims.
    Write a precise, universally accurate 1-sentence dictionary definition for each.
    Only define highly technical jargon or architecture-specific terms. Do not define general tech terms like
    'benchmarks' or 'latency' unless they are used in a unique way. Do NOT define common English phrases. And
    don't give too many definitions and overcrowd the section.
    ALSO - If the provided search results do not contain enough technical detail to define a term, do not guess based on your training data.
    Instead, define it based strictly on the provided context or mark it as [Technical Term - Context Limited].
    
    Output ONLY a valid JSON object matching this exact schema:
    {{
        "keep_ids": [1, 3],
        "definitions": {{"Jargon Word": "The definition."}}
    }}
    """
    try:
        raw_response = ask_gemini_gatekeeper(prompt, response_format="json_object")
        editor_logic = json.loads(raw_response)
        
        # 2. PYTHON STITCHING: Rebuild the array using the original untouched objects!
        keep_ids = editor_logic.get("keep_ids", [])
        
        # Filter the original array based on Gemini's decisions
        deduped_sources = [item for item in claims_data if item["id"] in keep_ids]
        
        # Safety net: if Gemini hallucinates or empties the list, fallback to all claims
        if not deduped_sources:
            deduped_sources = claims_data
            
        return {
            "sources": deduped_sources,
            "definitions": editor_logic.get("definitions", {})
        }
        
    except Exception as e:
        print(f"⚠️ Gemini Editor Error: {e}")
        # Absolute safety net: return the original format untouched
        return {
            "sources": claims_data, 
            "definitions": {}
        }