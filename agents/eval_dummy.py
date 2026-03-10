"""
Agent3: Evaluation + SEO Agent (DUMMY IMPLEMENTATION)

Replace this dummy with real Agent3 later.
Real implementation should:
- Score article quality (factual, SEO, readability)
- Validate claims against research data
- Check SEO optimization
- Return remarks and suggestions (NOT rewrites)
- Agent2 remains the only generator
"""

from state import PipelineState, Evaluation


def evaluation_agent_node(state: PipelineState) -> dict:
    """
    Evaluation Agent - scores draft and provides feedback.
    
    INPUT CONTRACT:
        - state["draft_article"]: DraftArticle
        - state["research_data"]["claims"]: List[Claim]
        - state["target_keyword"]: Optional[str]
    
    OUTPUT CONTRACT:
        - Evaluation with scores, status, remarks, rewrite_suggestions
    
    IMPORTANT:
        Agent3 DOES NOT rewrite the article.
        Agent3 only provides feedback.
        Agent2 remains the only generator.
    
    DUMMY BEHAVIOR:
        Returns NEEDS_REVISION on first pass,
        Returns APPROVED on second pass (or if revision_count >= 2).
    """
    
    draft = state["draft_article"]
    research_data = state["research_data"]
    target_keyword = state.get("target_keyword", "NVIDIA B200")
    revision_count = state["revision_count"]
    
    print(f"[EVAL AGENT] Evaluating article: '{draft['title']}'")
    print(f"[EVAL AGENT] Current revision count: {revision_count}")
    
    # Replace this dummy with real AgentX later
    
    # Simulate evaluation logic:
    # - First pass: request revision
    # - Second pass or max revisions: approve
    
    if revision_count == 0:
        # First evaluation - request improvements
        mock_evaluation: Evaluation = {
            "scores": {
                "factual": 0.85,
                "seo": 0.70,
                "readability": 0.80
            },
            "status": "NEEDS_REVISION",
            "remarks": [
                "SEO score is below target threshold (0.70 < 0.85)",
                "Keyword density for 'NVIDIA B200' could be improved",
                "Consider adding more technical depth in the architecture section"
            ],
            "rewrite_suggestions": [
                "Increase keyword usage in subheadings",
                "Add more specific technical specifications",
                "Expand the performance comparison section with concrete metrics"
            ]
        }
        print("[EVAL AGENT] Status: NEEDS_REVISION")
        
    else:
        # Revision pass or max revisions reached - approve
        mock_evaluation: Evaluation = {
            "scores": {
                "factual": 0.92,
                "seo": 0.88,
                "readability": 0.85
            },
            "status": "APPROVED",
            "remarks": [
                "All quality thresholds met",
                "SEO optimization improved significantly",
                "Technical depth is appropriate for target audience"
            ],
            "rewrite_suggestions": []
        }
        print("[EVAL AGENT] Status: APPROVED")
    
    print(f"[EVAL AGENT] Scores - Factual: {mock_evaluation['scores']['factual']}, "
          f"SEO: {mock_evaluation['scores']['seo']}, "
          f"Readability: {mock_evaluation['scores']['readability']}")
    print(f"[EVAL AGENT] ✓ Checkpoint will be saved after this node")
    
    return {
        "evaluation": mock_evaluation
    }
