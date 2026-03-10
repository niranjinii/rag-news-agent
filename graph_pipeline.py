"""
LangGraph Orchestration Pipeline for Multi-Agent Article Writer.

This module defines the stateful execution graph with:
- Conditional routing
- Revision loops
- Modular agent nodes
- Clear separation of concerns
- INTEGRATED CHECKPOINTING for state persistence

The graph is agent-agnostic: agents can be swapped without modifying pipeline logic.
"""

from typing import Literal, Optional
from langgraph.graph import StateGraph, END
from state import PipelineState
from agents.research_dummy import research_agent_node
from agents.writer_dummy import writer_agent_node
from agents.eval_dummy import evaluation_agent_node
from checkpointing import get_checkpoint_manager


# ========================================
# CONSTANTS
# ========================================

MAX_REVISIONS = 2


# ========================================
# CONDITIONAL ROUTING LOGIC
# ========================================

def should_revise(state: PipelineState) -> Literal["writer", "end"]:
    """
    Conditional edge: decides whether to loop back to writer or end.
    
    ROUTING LOGIC:
    1. If evaluation status is "NEEDS_REVISION" AND revision_count < MAX_REVISIONS:
       → Route to writer node for revision
    2. Otherwise:
       → Route to END
    
    This function has NO knowledge of agent internals.
    It only inspects the state contract.
    """
    
    evaluation = state.get("evaluation")
    revision_count = state["revision_count"]
    
    if not evaluation:
        # Safety: if no evaluation exists, end the pipeline
        print("[ROUTER] No evaluation found, ending pipeline")
        return "end"
    
    status = evaluation["status"]
    
    if status == "NEEDS_REVISION" and revision_count < MAX_REVISIONS:
        print(f"[ROUTER] Status: NEEDS_REVISION, Revision count: {revision_count}/{MAX_REVISIONS}")
        print("[ROUTER] → Routing to WRITER for revision")
        return "writer"
    else:
        if revision_count >= MAX_REVISIONS:
            print(f"[ROUTER] Max revisions reached ({MAX_REVISIONS}), forcing approval")
        else:
            print(f"[ROUTER] Status: {status}")
        print("[ROUTER] → Routing to END")
        return "end"


def increment_revision_counter(state: PipelineState) -> dict:
    """
    Increment revision counter after evaluation.
    This runs before the conditional edge.
    """
    return {
        "revision_count": state["revision_count"] + 1
    }


# ========================================
# GRAPH CONSTRUCTION
# ========================================

def build_article_writer_graph() -> StateGraph:
    """
    Constructs the LangGraph StateGraph for the article writing pipeline.
    
    GRAPH FLOW:
    
        START
          ↓
       RESEARCH (Agent1) → [CHECKPOINT SAVED]
          ↓
       WRITER (Agent2) → [CHECKPOINT SAVED]
          ↓
       EVALUATION (Agent3) → [CHECKPOINT SAVED]
          ↓
       INCREMENT_COUNTER
          ↓
       [CONDITIONAL EDGE]
          ↓
         / \
        /   \
    WRITER  END
      ↑
      └─── (revision loop)
    
    NODES:
    - research: Agent1 - gathers research data
    - writer: Agent2 - generates article
    - evaluation: Agent3 - scores and evaluates
    - increment: Helper node to track revisions
    
    EDGES:
    - research → writer (sequential)
    - writer → evaluation (sequential)
    - evaluation → increment (sequential)
    - increment → [conditional] (branching)
    - conditional → writer OR end (revision loop or completion)
    
    CHECKPOINTING:
    When compiled with a checkpointer, state is automatically saved after each node.
    This enables resume-from-failure functionality.
    """
    
    # Initialize the graph with our state schema
    workflow = StateGraph(PipelineState)
    
    # ========================================
    # ADD NODES
    # ========================================
    
    # Agent nodes - fully replaceable
    workflow.add_node("research", research_agent_node)
    workflow.add_node("writer", writer_agent_node)
    workflow.add_node("evaluation", evaluation_agent_node)
    
    # Control flow node
    workflow.add_node("increment", increment_revision_counter)
    
    # ========================================
    # ADD EDGES
    # ========================================
    
    # Set entry point
    workflow.set_entry_point("research")
    
    # Sequential flow: research → writer → evaluation
    workflow.add_edge("research", "writer")
    workflow.add_edge("writer", "evaluation")
    workflow.add_edge("evaluation", "increment")
    
    # Conditional edge: revision loop or end
    workflow.add_conditional_edges(
        "increment",
        should_revise,
        {
            "writer": "writer",  # Loop back for revision
            "end": END           # Complete the pipeline
        }
    )
    
    print("[GRAPH] Pipeline graph constructed successfully")
    print(f"[GRAPH] Max revisions configured: {MAX_REVISIONS}")
    
    return workflow


# ========================================
# GRAPH COMPILATION WITH CHECKPOINTING
# ========================================

def compile_article_writer_graph(
    enable_checkpointing: bool = True,
    checkpoint_backend: str = "memory",  # Changed default to memory for compatibility
    checkpoint_db_path: str = "checkpoints.db"
):
    """
    Compiles the graph with integrated checkpointing support.
    
    CHECKPOINTING BENEFITS:
    - State saved after each node execution
    - Resume from exact failure point
    - No wasted LLM calls on retry
    - Full execution history for debugging
    - Automatic recovery support
    
    Args:
        enable_checkpointing: Enable automatic state persistence (default: True)
        checkpoint_backend: Backend type - "sqlite", "postgres", or "memory" (default: "memory")
        checkpoint_db_path: Path to checkpoint database (default: "checkpoints.db")
    
    Returns:
        Compiled LangGraph application with checkpointing enabled
    
    Example:
        # With memory checkpointing (works everywhere, but not persistent)
        app = compile_article_writer_graph(enable_checkpointing=True)
        
        # With SQLite checkpointing (persistent, if supported)
        app = compile_article_writer_graph(
            enable_checkpointing=True,
            checkpoint_backend="sqlite"
        )
        
        # Without checkpointing (for testing only)
        app = compile_article_writer_graph(enable_checkpointing=False)
    """
    
    workflow = build_article_writer_graph()
    
    if enable_checkpointing:
        # Get checkpoint manager and backend
        try:
            checkpoint_manager = get_checkpoint_manager(
                backend=checkpoint_backend,
                db_path=checkpoint_db_path,
                force_recreate=False  # Reuse existing manager
            )
            checkpointer = checkpoint_manager.get_checkpointer()
            
            # Verify checkpointer is valid
            if checkpointer is None:
                print("[GRAPH] ⚠ Checkpointer is None, compiling without checkpointing")
                app = workflow.compile()
            else:
                # Compile graph WITH checkpointing
                app = workflow.compile(checkpointer=checkpointer)
                
                print(f"[GRAPH] ✓ Checkpointing ENABLED (backend: {checkpoint_backend})")
                print(f"[GRAPH] ✓ State will be saved after each node")
                
                if checkpoint_backend == "sqlite":
                    print(f"[GRAPH] ✓ Checkpoints stored in: {checkpoint_db_path}")
                elif checkpoint_backend == "memory":
                    print(f"[GRAPH] ⚠ Using memory backend (not persistent between runs)")
                
                print(f"[GRAPH] ✓ Resume capability: ACTIVE")
                
        except Exception as e:
            print(f"[GRAPH] ⚠ Checkpointing setup failed: {e}")
            print(f"[GRAPH] ⚠ Compiling without checkpointing")
            app = workflow.compile()
    else:
        # Compile graph WITHOUT checkpointing (ephemeral execution)
        app = workflow.compile()
        
        print("[GRAPH] ⚠ Checkpointing DISABLED")
        print("[GRAPH] ⚠ Pipeline execution is ephemeral (no persistence)")
        print("[GRAPH] ⚠ Cannot resume from failures")
    
    print("[GRAPH] Pipeline compiled and ready for execution")
    
    return app


# ========================================
# CONVENIENCE FUNCTION
# ========================================

def create_initial_state(
    topic: str,
    persona: str = "Technical Journalist",
    word_count: int = 800,
    target_keyword: str = None
) -> PipelineState:
    """
    Creates initial pipeline state with user inputs.
    
    Args:
        topic: Article topic
        persona: Writer persona (default: "Technical Journalist")
        word_count: Target word count (default: 800)
        target_keyword: SEO target keyword (optional)
    
    Returns:
        PipelineState initialized with user inputs
    """
    
    return PipelineState(
        topic=topic,
        research_data=None,
        draft_article=None,
        evaluation=None,
        revision_count=0,
        persona=persona,
        word_count=word_count,
        target_keyword=target_keyword
    )
