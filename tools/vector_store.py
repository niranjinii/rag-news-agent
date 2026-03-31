import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Load the model globally so it stays warm in memory
print("Loading Cross-Encoder model...")
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_chunks(original_topic: str, chunks_data: list, top_k: int = 15) -> list:
    """
    Takes the loose RRF results and uses a Cross-Encoder to ruthlessly
    grade actual logical relevance to the user's prompt.
    """
    if not chunks_data:
        return []
        
    # Create the pairs for the model: (User's Query, Document Text)
    pairs = [[original_topic, item["chunk"]] for item in chunks_data]
    
    # The model reads both simultaneously and outputs a precise relevance score
    scores = reranker_model.predict(pairs)
    
    # Attach the new scores to our chunks
    for i, item in enumerate(chunks_data):
        item["cross_encoder_score"] = float(scores[i])
        
    # Sort them highest to lowest based on the NEW semantic score
    reranked = sorted(chunks_data, key=lambda x: x["cross_encoder_score"], reverse=True)
    
    # Slice off the absolute best ones to feed to Llama
    return reranked[:top_k]

def run_hybrid_search(subqueries, all_chunks, all_metadata):
    """
    Ranks chunks based on keyword matching AND Markdown Header metadata.
    """
    print(f"🧮 Running Hybrid Search on {len(all_chunks)} chunks...")
    scored_chunks = []
    
    # Flatten our subqueries into one massive keyword list
    target_keywords = " ".join(subqueries).lower().split()
    
    for chunk, metadata in zip(all_chunks, all_metadata):
        score = 0
        chunk_lower = chunk.lower()
        
        # 1. Base Text Score (Simple keyword matching in the paragraph)
        for word in target_keywords:
            if word in chunk_lower:
                score += 1
                
        # 2. THE METADATA BOOST (The Rank 2 Fix!)
        # If the target keywords are actually in the Markdown Header, give it a massive multiplier
        headers_text = str(metadata).lower()
        for word in target_keywords:
            if word in headers_text:
                score += 5  # 5x boost for being in the header!
                
        # Only keep chunks that have at least some relevance
        if score > 0:
            scored_chunks.append({
                "chunk": chunk,
                "metadata": metadata,
                "score": score
            })
            
    # Sort by the highest score
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    # Return the top 15 absolute best chunks to the LLM
    return scored_chunks[:20]