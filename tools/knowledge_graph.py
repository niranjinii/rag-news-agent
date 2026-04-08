import requests
import urllib.parse
import os
import json

def extract_core_entity(topic: str) -> str:
    """
    Uses Llama 3 8B (via Groq) to instantly extract the core product name.
    We use requests.post directly so we don't mess up your other LLM files.
    """
    api_key = os.environ.get("GROQ_API_KEY") 
    
    if not api_key:
        print("⚠️ No GROQ_API_KEY found in environment. Falling back to raw topic.")
        return topic

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"Extract ONLY the primary technical product, architecture, or company name from this search query: '{topic}'. Do not include conversational text, punctuation, or explanations. Output the exact name only."
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, # Zero creativity, strictly factual extraction
        "max_tokens": 15  
    }
    
    try:
        print(f"🧠 Asking Llama 8b to extract core entity from: '{topic}'...")
        # 3 second timeout
        response = requests.post(url, headers=headers, json=payload, timeout=3)
        response.raise_for_status()
        
        # Strip out any random quotes or spaces the LLM might add
        extracted_name = response.json()["choices"][0]["message"]["content"].strip().strip("'\"")
        print(f"🎯 Llama 8b extracted: '{extracted_name}'")
        
        return extracted_name
    except Exception as e:
        print(f"⚠️ 8b Extraction Failed: {e}. Falling back to algorithmic trimming.")
        return topic

def verify_entity_with_wikidata(topic: str) -> dict:
    """
    Pings Wikidata using a hybrid approach:
    1. First tries the LLM-extracted exact noun.
    2. Falls back to progressive algorithmic trimming if the LLM fails or is wrong.
    """
    
    # Let the 8b model try to find the pure entity first
    core_entity = extract_core_entity(topic)
    
    words = topic.split()
    
    # We build an array of attempts. 
    # [0] is the LLM's guess. The rest is our algorithmic safety net.
    search_attempts = [
        core_entity,
        topic, 
        " ".join(words[:4]) if len(words) > 4 else None,
        " ".join(words[:3]) if len(words) > 3 else None,
        " ".join(words[:2]) if len(words) > 2 else None
    ]
    
    # Filter out None values and remove duplicates (keeps order)
    seen = set()
    unique_attempts = []
    for s in search_attempts:
        if s and s.lower() not in seen:
            seen.add(s.lower())
            unique_attempts.append(s)

    headers = {
        'User-Agent': 'TechNewsroomBot/1.0 (University Engineering Project)'
    }

    for query in unique_attempts:
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={encoded_query}&language=en&format=json"
        
        try:
            print(f"🌐 KG Attempt: Searching for '{query}'...")
            response = requests.get(url, headers=headers, timeout=3)
            response.raise_for_status()
            data = response.json()

            if data.get('search'):
                top_result = data['search'][0]
                print(f"✅ KG Match Found: {top_result.get('id')} ({query})")
                return {
                    "kg_verified": True,
                    "kg_id": top_result.get('id'),
                    "kg_description": top_result.get('description', 'No description available')
                }
        except Exception as e:
            print(f"⚠️ KG Attempt Failed for '{query}': {e}")
            continue # Try the next fallback

    # If the loop finishes with no return
    return {
        "kg_verified": False,
        "kg_status": "Not found in global graph"
    }