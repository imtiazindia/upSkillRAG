"""
Data models for RAG engine components
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

class SearchType(Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"

class RetrievalStrategy(Enum):
    SINGLE_SHOT = "single_shot"
    MULTI_QUERY = "multi_query"
    HYBRID_RERANK = "hybrid_rerank"

class ResponseQuality(Enum):
    EXCELLENT = "excellent"  # Highly relevant, well-supported
    GOOD = "good"           # Relevant, adequate support
    FAIR = "fair"           # Somewhat relevant, limited support
    POOR = "poor"           # Irrelevant or unsupported

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with metadata"""
    content: str
    document_id: str
    chunk_id: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def citation_id(self) -> str:
        """Generate a unique citation identifier"""
        return f"{self.document_id}_{self.chunk_id}"

@dataclass
class RetrievedContext:
    """Collection of retrieved document chunks with scores"""
    chunks: List[DocumentChunk]
    search_scores: List[float]
    search_type: SearchType
    retrieval_strategy: RetrievalStrategy
    
    def get_top_chunks(self, top_k: int = 3) -> List[DocumentChunk]:
        """Get top K chunks by search score"""
        sorted_indices = sorted(range(len(self.search_scores)), 
                              key=lambda i: self.search_scores[i], reverse=True)
        return [self.chunks[i] for i in sorted_indices[:top_k]]
    
    @property
    def average_score(self) -> float:
        """Average search score of retrieved chunks"""
        return sum(self.search_scores) / len(self.search_scores) if self.search_scores else 0.0

@dataclass
class ConversationTurn:
    """A single turn in a conversation"""
    query: str
    response: str
    timestamp: datetime
    contexts_used: List[DocumentChunk]
    
    def to_context_string(self) -> str:
        """Convert conversation turn to context string for LLM"""
        return f"User: {self.query}\nAssistant: {self.response}"

@dataclass
class RAGQuery:
    """Enhanced query for RAG system"""
    original_query: str
    enhanced_queries: List[str]  # For multi-query retrieval
    conversation_history: List[ConversationTurn]
    search_type: SearchType = SearchType.HYBRID
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK
    
    @property
    def is_follow_up(self) -> bool:
        """Check if this is a follow-up question"""
        return len(self.conversation_history) > 0

@dataclass
class RAGResponse:
    """Complete RAG response with citations and quality assessment"""
    answer: str
    original_query: str
    contexts_used: List[DocumentChunk]
    citations: Dict[str, List[str]]  # citation_id -> list of answer segments
    quality: ResponseQuality
    confidence_score: float  # 0.0 to 1.0
    processing_time: float
    retrieval_metrics: Dict[str, Any]
    
    def format_with_citations(self) -> str:
        """Format response with separate citations section"""
        response_text = self.answer
        
        if self.citations:
            citation_section = "\n\n---\n**Sources:**\n"
            for citation_id, segments in self.citations.items():
                # Find the chunk for this citation
                chunk = next((c for c in self.contexts_used if c.citation_id == citation_id), None)
                if chunk:
                    source_info = f"- {chunk.document_id}"
                    if chunk.section_title:
                        source_info += f" - {chunk.section_title}"
                    if chunk.page_number:
                        source_info += f" (Page {chunk.page_number})"
                    
                    citation_section += f"{source_info}\n"
            
            response_text += citation_section
        
        return response_text

@dataclass
class RAGConfig:
    """Configuration for RAG engine"""
    # Retrieval settings
    top_k_retrieval: int = 10  # Initial retrieval count
    top_k_final: int = 3       # Final chunks to use
    similarity_threshold: float = 0.7
    enable_hybrid_search: bool = True
    
    # Query processing
    enable_query_expansion: bool = True
    max_conversation_turns: int = 10
    
    # Response generation
    enable_quality_check: bool = True
    min_confidence_threshold: float = 0.6
    
    # Performance
    enable_caching: bool = True
    cache_ttl: int = 3600

@dataclass
class RetrievalMetrics:
    """Metrics for retrieval performance"""
    retrieval_time: float
    chunks_considered: int
    chunks_retrieved: int
    avg_similarity_score: float
    search_types_used: List[SearchType]
    query_expansion_count: int

@dataclass
class QualityMetrics:
    """Metrics for response quality"""
    answer_relevance: float  # 0.0 to 1.0
    context_relevance: float  # 0.0 to 1.0
    citation_density: float  # Citations per sentence
    hallucination_score: float  # 0.0 to 1.0 (lower is better)
    coherence_score: float  # 0.0 to 1.0
