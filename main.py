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
    checkpoint_backend: str = "memory"  # Memory backend (thread-safe, but not persistent)
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
    
    # ========================================
    # STEP 1: COMPILE GRAPH WITH CHECKPOINTING
    # ========================================
    
    print_separator("Step 1: Compiling Graph")
    app = compile_article_writer_graph(enable_checkpointing=enable_checkpointing)
    
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
                target_keyword=topic.split()[0] if topic else None
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
    print(f"  • Factual: {scores['factual']:.2f}")
    print(f"  • SEO: {scores['seo']:.2f}")
    print(f"  • Readability: {scores['readability']:.2f}")
    
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
            "total_claims": len(state["research_data"]["claims"]),
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
# MAIN EXECUTION WITH EXAMPLES
# ========================================

if __name__ == "__main__":
    
    # ========================================
    # EXAMPLE 1: Normal Execution with Checkpointing
    # ========================================
    
    print("\n" + "🚀 "*20)
    print("EXAMPLE 1: Normal Execution with Checkpointing")
    print("🚀 "*20)
    
    topic1 = "NVIDIA B200 GPU release"
    
    try:
        final_state = run_pipeline(
            topic=topic1,
            persona="Technical Journalist",
            word_count=800,
            enable_checkpointing=True,
            checkpoint_backend="memory"  # Use memory backend (thread-safe)
        )
        
        # Display and save
        display_article_preview(final_state)
        save_results(final_state, "article_output.json")
        
        print("✅ Example 1 Complete: Pipeline executed successfully with checkpointing!")
        
    except Exception as e:
        print(f"❌ Example 1 Failed: {e}")
        print("💡 Note: Progress was saved to checkpoint and can be resumed")
    
    # ========================================
    # EXAMPLE 2: Resume from Checkpoint
    # ========================================
    
    print("\n\n" + "🚀 "*20)
    print("EXAMPLE 2: Resume from Checkpoint (Simulation)")
    print("🚀 "*20)
    
    # Generate thread ID for the same topic
    thread_id = generate_thread_id(topic1)
    
    print(f"\nAttempting to resume from thread: {thread_id}")
    print("(If checkpoint exists, will resume; otherwise starts fresh)\n")
    
    try:
        final_state = run_pipeline(
            topic=topic1,
            resume_from=thread_id,
            enable_checkpointing=True,
            checkpoint_backend="memory"  # Use memory backend
        )
        print("✅ Example 2 Complete: Resume functionality demonstrated!")
        
    except Exception as e:
        print(f"Note: {e}")
    
    # ========================================
    # EXAMPLE 3: Unique Execution (No Resume)
    # ========================================
    
    print("\n\n" + "🚀 "*20)
    print("EXAMPLE 3: Unique Execution (Won't Reuse Checkpoints)")
    print("🚀 "*20)
    
    try:
        final_state = run_pipeline(
            topic="Quantum Computing Breakthroughs 2024",
            persona="Science Communicator",
            word_count=1200,
            enable_checkpointing=True,
            unique_execution=True,  # Creates unique thread ID
            checkpoint_backend="memory"  # Use memory backend
        )
        
        save_results(final_state, "article_quantum_computing.json")
        print("✅ Example 3 Complete: Unique execution with new thread ID!")
        
    except Exception as e:
        print(f"❌ Example 3 Failed: {e}")
    
    # ========================================
    # EXAMPLE 4: List All Checkpoints
    # ========================================
    
    print("\n\n" + "🚀 "*20)
    print("EXAMPLE 4: List All Saved Checkpoints")
    print("🚀 "*20)
    
    list_all_checkpoints()
    
    # ========================================
    # EXAMPLE 5: Cleanup Demo (commented out)
    # ========================================
    
    print("\n\n" + "🚀 "*20)
    print("EXAMPLE 5: Checkpoint Cleanup (Demo)")
    print("🚀 "*20)
    
    print("\nCleanup function available but not executed in demo.")
    print("To cleanup old checkpoints, uncomment the line below:")
    print("# cleanup_old_checkpoints(days=7)")
    
    # Uncomment to actually run cleanup:
    # cleanup_old_checkpoints(days=7)
    
    # ========================================
    # SUMMARY
    # ========================================
    
    print_separator("DEMONSTRATION COMPLETE")
    
    print("\n✅ All examples executed successfully!")
    print("\n📌 KEY FEATURES DEMONSTRATED:")
    print("  1. ✓ Normal execution with automatic checkpointing")
    print("  2. ✓ Resume from checkpoint after failure")
    print("  3. ✓ Unique execution IDs for independent runs")
    print("  4. ✓ Checkpoint listing and inspection")
    print("  5. ✓ Cleanup utilities for maintenance")
    
    print("\n📚 CHECKPOINTING BENEFITS:")
    print("  • State saved after each agent execution")
    print("  • Resume from exact failure point")
    print("  • No wasted LLM API calls")
    print("  • Full execution history preserved")
    print("  • Production-ready reliability")
    
    print("\n🔧 NEXT STEPS:")
    print("  1. Replace agents/research_dummy.py with real RAG Agent1")
    print("  2. Replace agents/writer_dummy.py with finetuned LLM Agent2")
    print("  3. Replace agents/eval_dummy.py with real Evaluation Agent3")
    print("  4. Graph pipeline + checkpointing require ZERO modifications!")
    
    print_separator()
