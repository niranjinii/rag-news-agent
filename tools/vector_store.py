import chromadb
from rank_bm25 import BM25Okapi

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
    return scored_chunks[:15]