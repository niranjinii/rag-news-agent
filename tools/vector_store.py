import chromadb
from rank_bm25 import BM25Okapi

def run_hybrid_search(subqueries, all_chunks, all_metadata):
    """Indexes chunks into RAM and runs Reciprocal Rank Fusion (RRF)."""
    chroma_client = chromadb.Client()
    collection_name = "research_temp_graph"
    
    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass
        
    collection = chroma_client.create_collection(name=collection_name)
    
    # Add to Chroma
    collection.add(
        documents=all_chunks,
        metadatas=all_metadata,
        ids=[str(i) for i in range(len(all_chunks))]
    )
    
    # Initialize BM25
    tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    best_results = []
    k = min(5, len(all_chunks))
    
    for query in subqueries:
        vector_results = collection.query(query_texts=[query], n_results=k)
        vector_ids = vector_results['ids'][0]
        
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
        bm25_ids = [str(i) for i in bm25_top_indices]

        # RRF Math
        rrf_scores = {}
        for rank, doc_id in enumerate(vector_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (rank + 60))
        for rank, doc_id in enumerate(bm25_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (rank + 60))
            
        # --- THE UPGRADE: Grab Top 3 instead of Top 1 ---
        top_doc_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:3]
        
        for doc_id in top_doc_ids:
            best_chunk_idx = int(doc_id)
            best_results.append({
                "chunk": all_chunks[best_chunk_idx],
                "metadata": all_metadata[best_chunk_idx]
            })
            
    return best_results