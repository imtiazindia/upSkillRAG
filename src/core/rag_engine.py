"""
Main RAG engine orchestrator
Coordinates all components to provide complete RAG functionality
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .models import (
    RAGQuery, RAGResponse, ConversationTurn, DocumentChunk,
    RAGConfig, SearchType, RetrievalStrategy, ResponseQuality
)
from .query_processor import QueryProcessor
from .context_retriever import ContextRetriever
from .response_builder import ResponseBuilder
from .quality_checker import QualityChecker
from llm import LLMManager, LLMConfig, ModelType

logger = logging.getLogger(__name__)

class RAGEngine:
    """
    Main RAG engine that orchestrates all components
    Provides complete RAG functionality with conversation support
    """
    
    def __init__(self, 
                 vector_store: Any,  # Your vector store implementation
                 llm_config: Optional[LLMConfig] = None,
                 rag_config: Optional[RAGConfig] = None):
        
        # Initialize configurations
        self.llm_config = llm_config or LLMConfig(
            provider="ollama",
            model=ModelType.TINYLLAMA,
            base_url="http://localhost:11434"
        )
        
        self.rag_config = rag_config or RAGConfig()
        
        # Initialize LLM manager
        self.llm_manager = LLMManager(self.llm_config)
        
        # Initialize RAG components
        self.query_processor = QueryProcessor(
            self.llm_manager, 
            self.rag_config.enable_query_expansion
        )
        
        self.context_retriever = ContextRetriever(
            vector_store=vector_store,
            similarity_threshold=self.rag_config.similarity_threshold,
            top_k_retrieval=self.rag_config.top_k_retrieval,
            top_k_final=self.rag_config.top_k_final
        )
        
        self.response_builder = ResponseBuilder(
            self.llm_manager,
            enable_citations=True
        )
        
        self.quality_checker = QualityChecker(
            self.llm_manager,
            enable_llm_validation=self.rag_config.enable_quality_check
        )
        
        # Conversation state
        self.conversation_history: List[ConversationTurn] = []
        
        # Performance tracking
        self.total_queries_processed = 0
        self.average_processing_time = 0.0
        
        logger.info("RAG Engine initialized successfully")
    
    def initialize(self) -> bool:
        """
        Initialize the RAG engine and all components
        
        Returns:
            bool: True if initialization successful
        """
        logger.info("Initializing RAG Engine...")
        
        try:
            # Initialize LLM manager
            if not self.llm_manager.initialize():
                logger.error("LLM manager initialization failed")
                return False
            
            # Verify vector store connectivity
            # This would be specific to your vector store implementation
            logger.info("RAG Engine initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"RAG Engine initialization failed: {e}")
            return False
    
    def ask_question(self, 
                    question: str, 
                    search_type: SearchType = SearchType.HYBRID,
                    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK) -> RAGResponse:
        """
        Main method to ask a question and get a RAG response
        
        Args:
            question: User question
            search_type: Type of search to perform
            retrieval_strategy: Strategy for retrieval
            
        Returns:
            RAGResponse: Complete RAG response
        """
        start_time = time.time()
        self.total_queries_processed += 1
        
        logger.info(f"Processing question: '{question}'")
        
        try:
            # Step 1: Process the query
            rag_query = self.query_processor.process_query(
                user_query=question,
                conversation_history=self.conversation_history,
                search_type=search_type,
                retrieval_strategy=retrieval_strategy
            )
            
            # Step 2: Retrieve relevant context
            context, retrieval_metrics = self.context_retriever.retrieve_context(rag_query)
            
            # Step 3: Check if we have sufficient context
            if not self._has_sufficient_context(context, rag_query):
                return self._create_insufficient_context_response(question, retrieval_metrics)
            
            # Step 4: Build response using LLM and context
            response = self.response_builder.build_response(
                query=rag_query,
                context=context,
                conversation_history=self.conversation_history
            )
            
            # Step 5: Validate response quality
            quality_metrics = self.quality_checker.validate_response(
                response=response,
                query=rag_query,
                context=context
            )
            
            # Step 6: Decide whether to show the response
            should_show, reason = self.quality_checker.should_show_response(
                response=response,
                metrics=quality_metrics,
                min_confidence=self.rag_config.min_confidence_threshold
            )
            
            if not should_show:
                logger.warning(f"Response quality below threshold: {reason}")
                response = self._create_quality_warning_response(question, reason, response)
            
            # Step 7: Update conversation history
            self._update_conversation_history(rag_query, response, context.chunks)
            
            # Step 8: Update performance metrics
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time)
            
            logger.info(f"Question processed in {processing_time:.2f}s - Quality: {response.quality.value}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return self._create_error_response(question, str(e))
    
    def _has_sufficient_context(self, context: Any, query: RAGQuery) -> bool:
        """
        Check if we have sufficient context to answer the question
        
        Args:
            context: Retrieved context
            query: Processed query
            
        Returns:
            bool: True if sufficient context available
        """
        if not context or not context.chunks:
            logger.warning(f"No context retrieved for query: '{query.original_query}'")
            return False
        
        # Validate retrieval quality
        if not self.context_retriever.validate_retrieval_quality(context, query.original_query):
            logger.warning(f"Poor retrieval quality for query: '{query.original_query}'")
            return False
        
        return True
    
    def _create_insufficient_context_response(self, 
                                            question: str,
                                            retrieval_metrics: Any) -> RAGResponse:
        """
        Create response for insufficient context situations
        
        Args:
            question: Original question
            retrieval_metrics: Retrieval performance metrics
            
        Returns:
            RAGResponse: Insufficient context response
        """
        return RAGResponse(
            answer=(
                "I couldn't find sufficient relevant information in the provided documents "
                f"to answer your question: '{question}'. \n\n"
                "This could be because:\n"
                "• The documents don't contain information about this topic\n"
                "• The question might need rephrasing for better matching\n"
                "• The relevant information might be in a different format or section\n\n"
                "Please try rephrasing your question or ask about a different topic."
            ),
            original_query=question,
            contexts_used=[],
            citations={},
            quality=ResponseQuality.POOR,
            confidence_score=0.0,
            processing_time=0.0,
            retrieval_metrics={
                "error": "insufficient_context",
                "retrieval_metrics": retrieval_metrics.__dict__ if retrieval_metrics else {}
            }
        )
    
    def _create_quality_warning_response(self, 
                                       question: str, 
                                       reason: str,
                                       original_response: RAGResponse) -> RAGResponse:
        """
        Create response with quality warning
        
        Args:
            question: Original question
            reason: Quality issue reason
            original_response: Original response that failed quality check
            
        Returns:
            RAGResponse: Quality warning response
        """
        warning_message = (
            f"I have some information, but I'm not confident about its accuracy for your question: '{question}'\n\n"
            f"Reason: {reason}\n\n"
            "The response may contain:\n"
            "• Information not fully supported by the documents\n"
            "• Potential inaccuracies or incomplete information\n"
            "• General knowledge not specific to your documents\n\n"
            "Please verify this information with additional sources if it's critical."
        )
        
        # Create a modified response with the warning
        return RAGResponse(
            answer=warning_message + "\n\n---\nOriginal Response:\n" + original_response.answer,
            original_query=question,
            contexts_used=original_response.contexts_used,
            citations=original_response.citations,
            quality=ResponseQuality.POOR,
            confidence_score=original_response.confidence_score * 0.5,  # Reduce confidence
            processing_time=original_response.processing_time,
            retrieval_metrics=original_response.retrieval_metrics
        )
    
    def _create_error_response(self, question: str, error: str) -> RAGResponse:
        """
        Create error response
        
        Args:
            question: Original question
            error: Error message
            
        Returns:
            RAGResponse: Error response
        """
        return RAGResponse(
            answer=(
                "I encountered an error while processing your question. "
                "This might be a temporary issue. Please try again in a moment.\n\n"
                f"Question: {question}\n"
                f"Error: {error}"
            ),
            original_query=question,
            contexts_used=[],
            citations={},
            quality=ResponseQuality.POOR,
            confidence_score=0.0,
            processing_time=0.0,
            retrieval_metrics={"error": error}
        )
    
    def _update_conversation_history(self, 
                                   query: RAGQuery, 
                                   response: RAGResponse,
                                   contexts_used: List[DocumentChunk]) -> None:
        """
        Update conversation history with new turn
        
        Args:
            query: Processed query
            response: Generated response
            contexts_used: Context chunks used for response
        """
        conversation_turn = ConversationTurn(
            query=query.original_query,
            response=response.answer,
            timestamp=datetime.now(),
            contexts_used=contexts_used
        )
        
        self.conversation_history.append(conversation_turn)
        
        # Limit conversation history length
        if len(self.conversation_history) > self.rag_config.max_conversation_turns:
            self.conversation_history.pop(0)
        
        logger.debug(f"Conversation history updated: {len(self.conversation_history)} turns")
    
    def _update_performance_metrics(self, processing_time: float) -> None:
        """
        Update performance tracking metrics
        
        Args:
            processing_time: Time taken to process the query
        """
        # Update average processing time using exponential moving average
        if self.average_processing_time == 0:
            self.average_processing_time = processing_time
        else:
            alpha = 0.1  # Smoothing factor
            self.average_processing_time = (alpha * processing_time + 
                                          (1 - alpha) * self.average_processing_time)
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get summary of current conversation
        
        Returns:
            Dict with conversation summary
        """
        return {
            "total_turns": len(self.conversation_history),
            "current_turn": len(self.conversation_history),
            "topics_covered": list(set(turn.query for turn in self.conversation_history[-5:])),
            "average_response_quality": (
                sum(turn.contexts_used for turn in self.conversation_history) / 
                len(self.conversation_history) if self.conversation_history else 0
            )
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the RAG engine
        
        Returns:
            Dict with performance metrics
        """
        llm_metrics = self.llm_manager.get_performance_metrics()
        
        return {
            "total_queries_processed": self.total_queries_processed,
            "average_processing_time": self.average_processing_time,
            "conversation_turns": len(self.conversation_history),
            "llm_performance": llm_metrics,
            "rag_config": {
                "search_type": self.rag_config.enable_hybrid_search,
                "query_expansion": self.rag_config.enable_query_expansion,
                "quality_check": self.rag_config.enable_quality_check,
                "similarity_threshold": self.rag_config.similarity_threshold
            }
        }
    
    def update_config(self, new_config: RAGConfig) -> None:
        """
        Update RAG configuration
        
        Args:
            new_config: New RAG configuration
        """
        self.rag_config = new_config
        
        # Update components with new config
        self.query_processor.enable_query_expansion = new_config.enable_query_expansion
        self.context_retriever.similarity_threshold = new_config.similarity_threshold
        self.context_retriever.top_k_final = new_config.top_k_final
        self.quality_checker.enable_llm_validation = new_config.enable_quality_check
        
        logger.info("RAG configuration updated")
    
    def shutdown(self) -> None:
        """Gracefully shutdown the RAG engine"""
        self.llm_manager.shutdown()
        self.conversation_history.clear()
        logger.info("RAG Engine shutdown complete")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


class SimpleRAGEngine:
    """
    Simplified RAG engine for basic use cases
    Provides a simpler interface for common RAG operations
    """
    
    def __init__(self, vector_store: Any, model: ModelType = ModelType.TINYLLAMA):
        self.vector_store = vector_store
        self.model = model
        
        # Initialize with default configurations
        llm_config = LLMConfig(model=model)
        rag_config = RAGConfig()
        
        self.engine = RAGEngine(vector_store, llm_config, rag_config)
    
    def initialize(self) -> bool:
        """Initialize the simple RAG engine"""
        return self.engine.initialize()
    
    def ask(self, question: str) -> str:
        """
        Simple ask method that returns just the answer text
        
        Args:
            question: User question
            
        Returns:
            str: Answer text
        """
        response = self.engine.ask_question(question)
        return response.answer
    
    def ask_with_sources(self, question: str) -> Tuple[str, List[str]]:
        """
        Ask question and return answer with source information
        
        Args:
            question: User question
            
        Returns:
            Tuple of (answer, list of source identifiers)
        """
        response = self.engine.ask_question(question)
        sources = [chunk.document_id for chunk in response.contexts_used]
        return response.answer, sources
    
    def clear_memory(self) -> None:
        """Clear conversation memory"""
        self.engine.clear_conversation_history()
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.engine.shutdown()
