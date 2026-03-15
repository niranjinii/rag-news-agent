"""
Evaluate Agent3 (dummy) using precomputed Agent1 and Agent2 sample JSON files.

This preserves the workflow and graph execution while injecting file outputs.
Revisions are disabled for this mode so evaluation runs once.
"""

from main import run_pipeline

print("=" * 70)
print("AGENT3 SAMPLE EVALUATION (INJECTED AGENT1 + AGENT2)")
print("=" * 70)

try:
    final_state = run_pipeline(
        topic="M4 Pro vs M4 Max",
        persona="Technical Journalist",
        word_count=900,
        enable_checkpointing=False,
        use_injected_inputs=True,
        agent1_file="agent1_output.json",
        agent2_file="agent2_output.json",
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Status: {final_state['evaluation']['status']}")
    print(f"Scores: {final_state['evaluation']['scores']}")
    print(f"Revisions: {final_state['revision_count']} (expected 1 due to post-eval increment)")

except Exception as error:
    print(f"\nERROR: {error}")
    raise
