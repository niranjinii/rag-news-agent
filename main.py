"""
Main runner for the Multi-Agent Article Writer Pipeline with Checkpointing.

This demonstrates how to:
1. Initialize the pipeline with checkpointing
2. Create initial state
3. Execute the graph with automatic state persistence
4. Resume from checkpoints after failures
5. Manage and inspect checkpoints
"""

import json
from typing import Optional
from graph_pipeline import compile_article_writer_graph, create_initial_state
from checkpointing import generate_thread_id, get_checkpoint_manager, get_thread_id_with_timestamp
from adapters.injected_state import load_research_data_from_file, load_draft_article_from_file


def print_separator(title: str = ""):
    """Print a visual separator for console output"""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print('='*70)
    else:
        print('='*70)


def run_pipeline(
    topic: str,
    persona: str = "Technical Journalist",
    word_count: int = 800,
    resume_from: Optional[str] = None,
    enable_checkpointing: bool = True,
    unique_execution: bool = False,
    checkpoint_backend: str = "memory",  # Memory backend (thread-safe, but not persistent)
    use_injected_inputs: bool = False,
    agent1_file: Optional[str] = None,
    agent2_file: Optional[str] = None,
):
    """
    Executes the complete article writing pipeline with checkpointing support.
    
    Args:
        topic: Article topic
        persona: Writer persona
        word_count: Target word count
        resume_from: Thread ID to resume from (None = start fresh)
        enable_checkpointing: Enable state persistence (default: True)
        unique_execution: Create unique thread ID even for same topic (default: False)
        checkpoint_backend: Checkpoint backend ("memory", "sqlite", "postgres")
        use_injected_inputs: If True, loads Agent1/Agent2 outputs from files and sends them to Agent3
        agent1_file: JSON file path for Agent1 output (required when use_injected_inputs=True)
        agent2_file: JSON file path for Agent2 output (required when use_injected_inputs=True)
    
    Returns:
        Final pipeline state
    
    Examples:
        # Normal execution with checkpointing
        state = run_pipeline("NVIDIA B200 GPU")
        
        # Resume from checkpoint
        state = run_pipeline("NVIDIA B200 GPU", resume_from="article_abc123")
        
        # Unique execution (won't reuse checkpoints)
        state = run_pipeline("NVIDIA B200 GPU", unique_execution=True)
    """
    
    print_separator("ARTICLE WRITER PIPELINE - STARTING")
    print(f"Topic: {topic}")
    print(f"Persona: {persona}")
    print(f"Target Word Count: {word_count}")
    print(f"Checkpointing: {'ENABLED ✓' if enable_checkpointing else 'DISABLED ✗'}")
    print(f"Injected Inputs: {'ENABLED ✓' if use_injected_inputs else 'DISABLED ✗'}")

    if use_injected_inputs and resume_from:
        raise ValueError("Cannot use resume_from with injected inputs. Start a fresh injected run.")

    if use_injected_inputs and (not agent1_file or not agent2_file):
        raise ValueError(
            "Injected mode requires both agent1_file and agent2_file. "
            "Example: run_pipeline(..., use_injected_inputs=True, agent1_file='agent1_output.json', agent2_file='agent2_output.json')"
        )

    preloaded_research_data = None
    preloaded_draft_article = None
    if use_injected_inputs:
        print(f"📥 Loading Agent1 sample from: {agent1_file}")
        print(f"📥 Loading Agent2 sample from: {agent2_file}")
        preloaded_research_data = load_research_data_from_file(agent1_file)
        preloaded_draft_article = load_draft_article_from_file(agent2_file)
        print("✓ Injected sample outputs loaded")
    
    # ========================================
    # STEP 1: COMPILE GRAPH WITH CHECKPOINTING
    # ========================================
    
    print_separator("Step 1: Compiling Graph")
    app = compile_article_writer_graph(
        enable_checkpointing=enable_checkpointing,
        checkpoint_backend=checkpoint_backend,
        preloaded_research_data=preloaded_research_data,
        preloaded_draft_article=preloaded_draft_article,
    )
    
    # ========================================
    # STEP 2: DETERMINE THREAD ID
    # ========================================
    
    print_separator("Step 2: Thread ID Configuration")
    
    if resume_from:
        # Resume from existing thread
        thread_id = resume_from
        print(f"🔄 RESUME MODE")
        print(f"   Thread ID: {thread_id}")
        
        if enable_checkpointing:
            # Check if checkpoint exists
            checkpoint_manager = get_checkpoint_manager()
            checkpoint_info = checkpoint_manager.get_checkpoint_info(thread_id)
            
            if checkpoint_info:
                print(f"   ✓ Checkpoint found: {checkpoint_info['checkpoint_id']}")
                print(f"   ✓ Will resume from last saved state")
            else:
                print(f"   ✗ No checkpoint found for this thread")
                print(f"   ⚠ Will start fresh execution")
                resume_from = None
        else:
            print("   ⚠ Checkpointing disabled, cannot resume")
            resume_from = None
    
    else:
        # New execution
        if unique_execution:
            thread_id = get_thread_id_with_timestamp(topic)
            print(f"🆕 NEW UNIQUE EXECUTION")
        else:
            thread_id = generate_thread_id(topic)
            print(f"🆕 NEW EXECUTION (deterministic thread ID)")
        
        print(f"   Thread ID: {thread_id}")
        
        if enable_checkpointing:
            print(f"   ✓ State will be checkpointed throughout execution")
            print(f"   ✓ Can resume with: resume_from='{thread_id}'")
    
    # ========================================
    # STEP 3: CREATE CONFIG FOR EXECUTION
    # ========================================
    
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    # ========================================
    # STEP 4: EXECUTE OR RESUME PIPELINE
    # ========================================
    
    print_separator("Step 3: Executing Pipeline")
    
    try:
        if resume_from and enable_checkpointing:
            # Resume from checkpoint
            print("📂 Loading checkpoint and resuming execution...")
            final_state = app.invoke(None, config=config)  # None = resume from checkpoint
        else:
            # Fresh execution
            print("🚀 Starting fresh execution...")
            initial_state = create_initial_state(
                topic=topic,
                persona=persona,
                word_count=word_count,
                target_keyword=topic.split()[0] if topic else None,
                disable_revisions=use_injected_inputs
            )
            
            final_state = app.invoke(initial_state, config=config)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        if enable_checkpointing:
            print(f"💾 Progress saved to checkpoint: {thread_id}")
            print(f"📌 Resume with: run_pipeline('{topic}', resume_from='{thread_id}')")
        raise
        
    except Exception as e:
        print_separator("❌ PIPELINE FAILED")
        print(f"Error: {type(e).__name__}: {e}")
        
        if enable_checkpointing:
            print(f"\n💾 State saved to checkpoint: {thread_id}")
            print(f"📌 Resume with: run_pipeline('{topic}', resume_from='{thread_id}')")
            print(f"📌 Or in code: run_pipeline(topic='{topic}', resume_from='{thread_id}')")
        else:
            print("\n⚠ Checkpointing was disabled - cannot resume")
        
        raise
    
    # ========================================
    # STEP 5: DISPLAY RESULTS
    # ========================================
    
    print_separator("✅ PIPELINE EXECUTION COMPLETE")
    
    print("\n📊 FINAL STATE SUMMARY:")
    print(f"  • Topic: {final_state['topic']}")
    print(f"  • Total Revisions: {final_state['revision_count']}")
    print(f"  • Final Status: {final_state['evaluation']['status']}")
    
    print("\n📈 EVALUATION SCORES:")
    scores = final_state['evaluation']['scores']
    accuracy_score = scores.get('accuracy', scores.get('factual', 0.0))
    citation_quality_score = scores.get('citation_quality', scores.get('seo', 0.0))
    readability_score = scores.get('readability', 0.0)
    coverage_score = scores.get('coverage', 0.0)
    claim_density_score = scores.get('claim_density', 0.0)
    print(f"  • Accuracy: {accuracy_score:.2f}")
    print(f"  • Citation Quality: {citation_quality_score:.2f}")
    print(f"  • Readability: {readability_score:.2f}")
    print(f"  • Coverage: {coverage_score:.2f}")
    print(f"  • Claim Density: {claim_density_score:.2f}")
    
    print("\n📝 ARTICLE DETAILS:")
    draft = final_state['draft_article']
    print(f"  • Title: {draft['title']}")
    print(f"  • Meta Description: {draft['meta_description'][:80]}...")
    print(f"  • Content Length: {len(draft['content_md'])} characters")
    print(f"  • Citations: {len(draft['citations'])} sources")
    
    print("\n💬 EVALUATION REMARKS:")
    for i, remark in enumerate(final_state['evaluation']['remarks'], 1):
        print(f"  {i}. {remark}")
    
    if enable_checkpointing:
        print(f"\n💾 CHECKPOINT INFO:")
        print(f"  • Thread ID: {thread_id}")
        print(f"  • All execution states saved")
        print(f"  • Can be inspected or deleted via checkpoint manager")
    
    print_separator()
    
    return final_state


def list_all_checkpoints():
    """List all saved pipeline executions with checkpoint information"""
    
    print_separator("SAVED PIPELINE CHECKPOINTS")
    
    checkpoint_manager = get_checkpoint_manager()
    threads = checkpoint_manager.list_threads(limit=50)
    
    if not threads:
        print("\n📭 No checkpoints found.")
        print("   Checkpoints are created when pipelines run with checkpointing enabled.")
        return
    
    print(f"\n📦 Found {len(threads)} pipeline execution(s):\n")
    
    for i, (thread_id, latest_checkpoint) in enumerate(threads, 1):
        print(f"{i}. Thread ID: {thread_id}")
        print(f"   Latest Checkpoint: {latest_checkpoint}")
        
        # Get detailed info
        info = checkpoint_manager.get_checkpoint_info(thread_id)
        if info and info.get('created_at'):
            print(f"   Created: {info['created_at']}")
        
        print()


def cleanup_old_checkpoints(days: int = 7):
    """Remove checkpoints older than specified days"""
    
    print_separator(f"CLEANUP: Checkpoints Older Than {days} Days")
    
    checkpoint_manager = get_checkpoint_manager()
    deleted = checkpoint_manager.cleanup_old_checkpoints(days=days)
    
    print(f"\n✓ Cleanup complete: {deleted} checkpoint(s) removed")


def delete_checkpoint(thread_id: str):
    """Delete a specific checkpoint thread"""
    
    print_separator(f"DELETE CHECKPOINT: {thread_id}")
    
    checkpoint_manager = get_checkpoint_manager()
    success = checkpoint_manager.delete_thread(thread_id)
    
    if success:
        print(f"\n✓ Checkpoint deleted: {thread_id}")
    else:
        print(f"\n✗ Failed to delete checkpoint: {thread_id}")


def save_results(state, output_file: str = "article_output.json"):
    """
    Saves the final article and metadata to a JSON file.
    
    Args:
        state: Final pipeline state
        output_file: Output filename
    """
    
    output_data = {
        "metadata": {
            "topic": state["topic"],
            "persona": state["persona"],
            "word_count": state["word_count"],
            "revision_count": state["revision_count"],
            "final_status": state["evaluation"]["status"]
        },
        "article": state["draft_article"],
        "evaluation": state["evaluation"],
        "research_summary": {
            "total_claims": len(state["research_data"].get("sources", [])),
            "total_sources": len(state["research_data"]["sources"])
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")


def display_article_preview(state):
    """
    Displays a preview of the generated article.
    
    Args:
        state: Final pipeline state
    """
    
    draft = state["draft_article"]
    
    print_separator("ARTICLE PREVIEW")
    print(f"\nTitle: {draft['title']}")
    print(f"\nMeta Description:\n{draft['meta_description']}")
    print(f"\nContent Preview (first 500 chars):\n{draft['content_md'][:500]}...")
    print(f"\n\nCitations:")
    for citation in draft['citations']:
        print(f"  • {citation}")
    print_separator()


# ========================================
# INTERACTIVE TERMINAL ENTRYPOINT
# ========================================

def _prompt_with_default(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value if value else default


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    value = input(f"{prompt} ({suffix}): ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes", "1", "true"}


def _prompt_int(prompt: str, default: int) -> int:
    while True:
        raw_value = input(f"{prompt} [{default}]: ").strip()
        if not raw_value:
            return default
        try:
            return int(raw_value)
        except ValueError:
            print("Please enter a valid integer.")


def interactive_main() -> None:
    print_separator("INTERACTIVE PIPELINE RUNNER")
    print("Provide the topic/title, then the full pipeline will run:")
    print("Research Agent → Writer Agent → Evaluator Agent\n")

    topic = ""
    while not topic:
        topic = input("Enter search title/topic: ").strip()
        if not topic:
            print("Topic cannot be empty.")

    persona = _prompt_with_default("Persona", "Technical Journalist")
    word_count = _prompt_int("Target word count", 800)

    enable_checkpointing = _prompt_yes_no("Enable checkpointing", True)
    checkpoint_backend = "memory"
    if enable_checkpointing:
        while True:
            backend_value = _prompt_with_default("Checkpoint backend (memory/sqlite/postgres)", "memory").lower()
            if backend_value in {"memory", "sqlite", "postgres"}:
                checkpoint_backend = backend_value
                break
            print("Please choose one of: memory, sqlite, postgres.")

    unique_execution = _prompt_yes_no("Use unique execution ID", False)

    resume_from = ""
    if enable_checkpointing:
        resume_from = input("Resume from existing thread_id (leave blank for fresh run): ").strip()

    use_injected_inputs = _prompt_yes_no("Use injected Agent1 + Agent2 JSON files", False)
    agent1_file = None
    agent2_file = None
    if use_injected_inputs:
        agent1_file = _prompt_with_default("Agent1 JSON file", "agent1_output.json")
        agent2_file = _prompt_with_default("Agent2 JSON file", "agent2_output.json")

    final_state = run_pipeline(
        topic=topic,
        persona=persona,
        word_count=word_count,
        resume_from=resume_from or None,
        enable_checkpointing=enable_checkpointing,
        unique_execution=unique_execution,
        checkpoint_backend=checkpoint_backend,
        use_injected_inputs=use_injected_inputs,
        agent1_file=agent1_file,
        agent2_file=agent2_file,
    )

    if _prompt_yes_no("Show article preview", True):
        display_article_preview(final_state)

    if _prompt_yes_no("Save output JSON", True):
        output_file = _prompt_with_default("Output filename", "article_output.json")
        save_results(final_state, output_file)

    print("\nRun complete.")


if __name__ == "__main__":
    interactive_main()
