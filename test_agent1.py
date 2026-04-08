import json
from state import PipelineState
from agents.research_agent import research_agent_node

def main():
    print("TESTING AGENT 1 (RESEARCHER) IN ISOLATION...")
    
    # 1. Create a dummy state that looks exactly like what LangGraph will pass in
    mock_state = PipelineState(
        topic="NVIDIA GeForce RTX 6090 Ti Founders Edition official thermal architecture specs", # Change this to whatever you want
        research_data=None,
        draft_article=None,
        evaluation=None,
        revision_count=0,
        persona="Technical Journalist",
        word_count=800,
        target_keyword=None
    )

    # 2. Run your agent directly
    try:
        output_state = research_agent_node(mock_state)
        
        # 3. Print the results beautifully
        print("\n" + "="*50)
        print("AGENT 1 OUTPUT:")
        print("="*50)
        print(json.dumps(output_state, indent=2))
        
    except Exception as e:
        print(f"\n ERROR: {e}")

if __name__ == "__main__":
    main()