"""
Agents package for the Multi-Agent Article Writer.

Each agent module provides a node function compatible with LangGraph.
All agents follow strict I/O contracts defined in state.py.

Agents are fully replaceable - swap implementations without touching graph logic.
"""

from .research_agent import research_agent_node
from .writer_agent import writer_agent_node
from .eval_dummy import evaluation_agent_node

__all__ = [
    'research_agent_node',
    'writer_agent_node',
    'evaluation_agent_node',
]
