"""
State schema for the multi-agent article writing pipeline.
Uses TypedDict for type safety and clear contracts.
"""

from typing import TypedDict, List, Dict, Optional, Literal, NotRequired

class VerifiedSource(TypedDict):
    """Self-contained fact package for the Writer Agent"""
    id: int
    subtopic: str
    extracted_claim: str
    raw_chunk: str
    url: str


class Claim(TypedDict):
    """Individual factual claim from research"""
    claim: str
    source_id: int


class Definition(TypedDict):
    """Term definition from research"""
    term: str
    definition: str


class Chunk(TypedDict):
    """RAG chunk from research"""
    source_id: int
    chunk: str


class Source(TypedDict):
    """Source metadata"""
    id: int
    url: str


class ResearchData(TypedDict):
    """Agent1 output schema"""
    definitions: Dict[str, str]
    sources: List[VerifiedSource]


class DraftArticle(TypedDict):
    """Agent2 output schema - structured Markdown"""
    title: str
    meta_description: str
    content_md: str
    citations: List[str]


class Scores(TypedDict):
    """Quality scores from Agent3"""
    accuracy: float
    citation_quality: float
    readability: float
    factual: NotRequired[float]
    seo: NotRequired[float]


class Evaluation(TypedDict):
    """Agent3 output schema"""
    scores: Scores
    status: Literal["APPROVED", "NEEDS_REVISION"]
    remarks: List[str]
    rewrite_suggestions: List[str]


class PipelineState(TypedDict):
    """
    Complete state for the article writing pipeline.
    This is the single source of truth passed between all nodes.
    """
    # Input
    topic: str
    
    # Agent1 output
    research_data: Optional[ResearchData]
    
    # Agent2 output
    draft_article: Optional[DraftArticle]
    
    # Agent3 output
    evaluation: Optional[Evaluation]
    
    # Control flow
    revision_count: int
    
    # Optional metadata
    persona: str
    word_count: int
    target_keyword: Optional[str]
    disable_revisions: NotRequired[bool]
