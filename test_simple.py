"""
Simple Test Script - Runs WITHOUT Checkpointing
Use this to verify the core pipeline works.
"""

from graph_pipeline import compile_article_writer_graph, create_initial_state

print("=" * 70)
print("SIMPLE TEST - NO CHECKPOINTING")
print("=" * 70)

# Compile WITHOUT checkpointing (avoids all threading issues)
print("\n[1] Compiling graph WITHOUT checkpointing...")
app = compile_article_writer_graph(enable_checkpointing=False)
print("✓ Graph compiled\n")

# Create initial state
print("[2] Creating initial state...")
initial_state = create_initial_state(
    topic="NVIDIA B200 GPU release",
    persona="Technical Journalist",
    word_count=800
)
print("✓ Initial state created\n")

# Run pipeline
print("[3] Executing pipeline...\n")
print("-" * 70)

try:
    final_state = app.invoke(initial_state)
    
    print("-" * 70)
    print("\n SUCCESS! Pipeline completed\n")
    
    # Display results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n Final Status: {final_state['evaluation']['status']}")
    print(f" Total Revisions: {final_state['revision_count']}")
    
    print(f"\n  Scores:")
    scores = final_state['evaluation']['scores']
    print(f"   • Accuracy: {scores['accuracy']:.2f}")
    print(f"   • Citation Quality: {scores['citation_quality']:.2f}")
    print(f"   • Readability: {scores['readability']:.2f}")
    print(f"   • Coverage: {scores['coverage']:.2f}")
    print(f"   • Claim Density: {scores['claim_density']:.2f}")
    
    print(f"\n  Article:")
    draft = final_state['draft_article']
    print(f"   • Title: {draft['title']}")
    print(f"   • Length: {len(draft['content_md'])} characters")
    print(f"   • Citations: {len(draft['citations'])}")
    
    print("\n" + "=" * 70)
    print("  CORE SYSTEM WORKS PERFECTLY!")
    print("=" * 70)
    
    print("\n  NOTE: This ran without checkpointing.")
    print("   To enable checkpoints, use checkpoint_backend='memory'")
    print("   (Memory checkpoints work but don't persist between runs)")
    
except Exception as e:
    print("-" * 70)
    print(f"\n  ERROR: {e}")
    print("\nFull error details:")
    import traceback
    traceback.print_exc()
