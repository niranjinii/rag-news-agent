import os
import json
import requests
from groq import Groq
from dotenv import load_dotenv

import os
import json
import requests
from groq import Groq
from dotenv import load_dotenv, find_dotenv

# 1. Force Python to hunt down the .env file wherever it is
env_path = find_dotenv()
print(f"DEBUG: Found .env file at: {env_path}")
load_dotenv(env_path)

# 2. Grab the keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# 3. Safety check so it doesn't crash blindly
if not GROQ_API_KEY:
    raise ValueError("STOP: GROQ_API_KEY is completely empty! Check your .env file.")

# Initialize the Groq client
client = Groq(api_key=GROQ_API_KEY)

def ask_llm(prompt, response_format="text"):
    """Global wrapper for Groq LLM calls."""
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": "llama-3.1-8b-instant", "messages": messages, "temperature": 0.2}
    if response_format == "json_object":
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

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
        response = ask_llm(prompt, response_format="json_object")
        return json.loads(response).get("queries", [topic])
    except:
        return [topic]

def google_search(query):
    """Fetches valid URLs using Serper API, avoiding garbage domains."""
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": 3}) 
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    blacklist = ["reddit.com", "quora.com", "youtube.com", "facebook.com", "twitter.com", "x.com"]
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        results = response.json()
        valid_urls = []
        for item in results.get('organic', []):
            if 'link' in item and not any(bad in item['link'] for bad in blacklist):
                valid_urls.append(item['link'])
                if len(valid_urls) == 3: # Laptop Saver: Grab top 1 valid URL
                    break
        return valid_urls
    except Exception as e:
        print(f"Serper API error: {e}")
        return []