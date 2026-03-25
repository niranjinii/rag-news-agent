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
            "model": "llama-3.3-70b-versatile",
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
    
def get_definitions_from_gemini(claims_data):
    """
    The Enrichment Step: Uses Gemini's up-to-date knowledge to define jargon 
    found in Llama's extracted claims.
    """
    # Combine all the claims into one readable paragraph for Gemini
    claims_text = "\n".join([item.get("extracted_claim", "") for item in claims_data])
    
    prompt = f"""
    You are a Technical Dictionary Editor. Read the following research claims:
    
    {claims_text}
    
    Identify the 2-3 most complex technical terms or jargon (e.g., 'M3 Max', 'SoC', 'Neural Engine').
    Provide a precise, universally accurate 1-sentence dictionary definition for each using your internal knowledge.
    
    Output ONLY a valid JSON object where keys are the terms and values are the definitions.
    Example: {{"M3 Max": "A high-performance custom Apple Silicon system-on-a-chip featuring a 40-core GPU."}}
    """
    try:
        # Call Gemini in JSON mode!
        raw_response = ask_gemini_gatekeeper(prompt, response_format="json_object")
        return json.loads(raw_response)
    except Exception as e:
        print(f"⚠️ Gemini Enrichment Error: {e}")
        return {}