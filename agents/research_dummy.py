"""
Agent1: Research Agent (DUMMY IMPLEMENTATION)

Replace this dummy with real Agent1 later.
Real implementation should:
- Perform RAG retrieval
- Extract claims from documents
- Generate definitions
- Return top relevant chunks
"""

from state import PipelineState, ResearchData


def research_agent_node(state: PipelineState) -> dict:
    """
    Research Agent - produces structured research data.
    
    INPUT CONTRACT:
        - state["topic"]: str
    
    OUTPUT CONTRACT:
        - ResearchData with claims, definitions, chunks, sources
    
    DUMMY BEHAVIOR:
        Returns static mock data following exact schema.
    """
    
    topic = state["topic"]
    
    # Replace this dummy with real AgentX later
    mock_research_data: ResearchData = {
        "claims": [
            {
                "claim": f"The {topic} represents a significant advancement in GPU technology",
                "source_id": 1
            },
            {
                "claim": "NVIDIA B200 features enhanced Tensor Cores for AI workloads",
                "source_id": 2
            },
            {
                "claim": "Expected performance improvements of 2.5x over previous generation",
                "source_id": 1
            }
        ],
        "definitions": [
            {
                "term": "Tensor Core",
                "definition": "Specialized hardware units designed for matrix multiplication operations in deep learning"
            },
            {
                "term": "HBM3e",
                "definition": "High Bandwidth Memory 3e - advanced memory technology offering increased bandwidth"
            }
        ],
        "top_chunks": [
            {
                "source_id": 1,
                "chunk": "NVIDIA announced the B200 GPU at GTC 2024, showcasing revolutionary architecture improvements..."
            },
            {
                "source_id": 2,
                "chunk": "The B200 GPU features 208 billion transistors and supports up to 192GB of HBM3e memory..."
            },
            {
                "source_id": 3,
                "chunk": "Early benchmarks suggest the B200 delivers exceptional performance in large language model training..."
            }
        ],
        "sources": [
            {
                "id": 1,
                "url": "https://nvidia.com/b200-announcement"
            },
            {
                "id": 2,
                "url": "https://techreview.com/nvidia-b200-analysis"
            },
            {
                "id": 3,
                "url": "https://benchmarks.ai/b200-performance"
            }
        ]
    }
    
    print(f"[RESEARCH AGENT] Processing topic: {topic}")
    print(f"[RESEARCH AGENT] Found {len(mock_research_data['claims'])} claims")
    print(f"[RESEARCH AGENT] ✓ Checkpoint will be saved after this node")
    
    return {
        "research_data": mock_research_data
    }
