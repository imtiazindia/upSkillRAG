"""
Query processor for RAG system
Handles conversational query understanding, expansion, and enhancement
"""

import logging
import re
from typing import List, Optional
from datetime import datetime

from .models import RAGQuery, ConversationTurn, SearchType, RetrievalStrategy
from llm import LLMManager, LLMRequest, ModelType

logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Processes and enhances user queries for better retrieval
    Supports conversational context and query expansion
    """
    
    def __init__(self, llm_manager: LLMManager, enable_query_expansion: bool = True):
        self.llm_manager = llm_manager
        self.enable_query_expansion = enable_query_expansion
        
        # Patterns for follow-up detection
        self.follow_up_patterns = [
            r'^(what|how|where|when|who|why)\s+',
            r'^(can you|could you|would you)',
            r'^(tell me more|explain|elaborate)',
            r'^\w+(\s+\w+){0,3}\?$'  # Short questions
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.follow_up_patterns]
        
        logger.info("Query Processor initialized")
    
    def process_query(self, 
                     user_query: str, 
                     conversation_history: List[ConversationTurn] = None,
                     search_type: SearchType = SearchType.HYBRID,
                     retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK) -> RAGQuery:
        """
        Process user query with conversation context and enhancement
        
        Args:
            user_query: Original user question
            conversation_history: Previous conversation turns
            search_type: Type of search to perform
            retrieval_strategy: Strategy for retrieval
            
        Returns:
            RAGQuery: Enhanced query object
        """
        logger.info(f"Processing query: '{user_query}'")
        
        # Normalize conversation history
        history = conversation_history or []
        
        # Detect if this is a follow-up question
        is_follow_up = self._detect_follow_up_question(user_query, history)
        
        # Enhance query based on context
        enhanced_queries = self._enhance_query(user_query, history, is_follow_up)
        
        # Create RAG query object
        rag_query = RAGQuery(
            original_query=user_query,
            enhanced_queries=enhanced_queries,
            conversation_history=history,
            search_type=search_type,
            retrieval_strategy=retrieval_strategy
        )
        
        logger.debug(f"Query processed: {len(enhanced_queries)} enhanced versions created")
        return rag_query
    
    def _detect_follow_up_question(self, query: str, history: List[ConversationTurn]) -> bool:
        """
        Detect if the query is a follow-up question
        
        Args:
            query: Current user query
            history: Conversation history
            
        Returns:
            bool: True if follow-up question detected
        """
        if not history:
            return False
        
        # Check pattern matching
        query_lower = query.lower().strip()
        for pattern in self.compiled_patterns:
            if pattern.match(query_lower):
                return True
        
        # Check for pronouns and references that indicate follow-up
        follow_up_indicators = [
            'it', 'that', 'this', 'they', 'them', 'their',
            'the above', 'the mentioned', 'previous', 'earlier'
        ]
        
        if any(indicator in query_lower for indicator in follow_up_indicators):
            return True
        
        # Check if query is very short (likely follow-up)
        words = query_lower.split()
        if len(words) <= 4 and query_lower.endswith('?'):
            return True
        
        return False
    
    def _enhance_query(self, 
                      original_query: str, 
                      history: List[ConversationTurn],
                      is_follow_up: bool) -> List[str]:
        """
        Enhance and expand the query for better retrieval
        
        Args:
            original_query: Original user question
            history: Conversation history
            is_follow_up: Whether this is a follow-up question
            
        Returns:
            List of enhanced query versions
        """
        enhanced_queries = [original_query]
        
        if not self.enable_query_expansion:
            return enhanced_queries
        
        try:
            if is_follow_up and history:
                # For follow-up questions, incorporate context
                contextual_queries = self._expand_follow_up_query(original_query, history)
                enhanced_queries.extend(contextual_queries)
            else:
                # For new questions, generate alternative phrasings
                alternative_queries = self._generate_alternative_queries(original_query)
                enhanced_queries.extend(alternative_queries)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for query in enhanced_queries:
                if query not in seen:
                    seen.add(query)
                    unique_queries.append(query)
            
            return unique_queries
            
        except Exception as e:
            logger.warning(f"Query enhancement failed, using original only: {e}")
            return [original_query]
    
    def _expand_follow_up_query(self, query: str, history: List[ConversationTurn]) -> List[str]:
        """
        Expand follow-up queries with conversation context
        
        Args:
            query: Current follow-up query
            history: Conversation history
            
        Returns:
            List of context-enhanced queries
        """
        # Get the most recent conversation for context
        recent_context = history[-1].to_context_string() if history else ""
        
        prompt = f"""
        The user asked a follow-up question. The previous conversation was:
        {recent_context}
        
        Current follow-up question: {query}
        
        Please rephrase this follow-up question into 2-3 standalone questions that 
        include the necessary context from the previous conversation. Make sure each 
        rephrased question can be understood on its own.
        
        Return each question on a separate line without numbering.
        """
        
        try:
            llm_request = LLMRequest(
                prompt=prompt,
                model=ModelType.TINYLLAMA,
                temperature=0.3,
                max_tokens=200
            )
            
            response = self.llm_manager.generate_response(llm_request)
            enhanced_queries = self._parse_llm_response(response.content)
            
            logger.debug(f"Expanded follow-up query '{query}' to {len(enhanced_queries)} versions")
            return enhanced_queries
            
        except Exception as e:
            logger.warning(f"LLM-based follow-up expansion failed: {e}")
            # Fallback: simple context addition
            if history:
                last_topic = history[-1].query
                return [f"{last_topic} - {query}", f"Regarding previous discussion: {query}"]
            return []
    
    def _generate_alternative_queries(self, query: str) -> List[str]:
        """
        Generate alternative phrasings of the query
        
        Args:
            query: Original query
            
        Returns:
            List of alternative query phrasings
        """
        prompt = f"""
        Generate 2-3 alternative phrasings of this search query that might help 
        retrieve relevant information. Focus on different ways someone might ask 
        the same question.
        
        Original query: {query}
        
        Return each alternative phrasing on a separate line without numbering.
        Keep the alternatives concise and directly related to the original query.
        """
        
        try:
            llm_request = LLMRequest(
                prompt=prompt,
                model=ModelType.TINYLLAMA,
                temperature=0.4,
                max_tokens=150
            )
            
            response = self.llm_manager.generate_response(llm_request)
            alternative_queries = self._parse_llm_response(response.content)
            
            logger.debug(f"Generated {len(alternative_queries)} alternative phrasings for '{query}'")
            return alternative_queries
            
        except Exception as e:
            logger.warning(f"LLM-based query expansion failed: {e}")
            # Fallback: simple variations
            return self._generate_simple_variations(query)
    
    def _generate_simple_variations(self, query: str) -> List[str]:
        """
        Generate simple query variations without LLM
        
        Args:
            query: Original query
            
        Returns:
            List of simple variations
        """
        variations = []
        query_lower = query.lower()
        
        # Add question mark if not present
        if not query_lower.endswith('?') and len(query.split()) <= 8:
            variations.append(query + '?')
        
        # Remove question mark for statement form
        if query_lower.endswith('?'):
            variations.append(query[:-1])
        
        # Add "what is" prefix for noun phrases
        words = query_lower.split()
        if len(words) <= 4 and not query_lower.startswith(('what', 'how', 'when', 'where', 'who', 'why')):
            variations.append(f"what is {query}")
        
        return variations
    
    def _parse_llm_response(self, response: str) -> List[str]:
        """
        Parse LLM response into individual queries
        
        Args:
            response: LLM response text
            
        Returns:
            List of parsed queries
        """
        lines = response.strip().split('\n')
        queries = []
        
        for line in lines:
            line = line.strip()
            # Remove numbering and bullets
            line = re.sub(r'^[\d\-•\.]\s*', '', line)
            # Remove quotes
            line = line.strip('"\'')
            
            if line and len(line) > 5:  # Reasonable length check
                queries.append(line)
        
        return queries
    
    def get_conversation_summary(self, history: List[ConversationTurn], max_turns: int = 5) -> str:
        """
        Generate a summary of conversation history for context
        
        Args:
            history: Conversation history
            max_turns: Maximum number of recent turns to include
            
        Returns:
            str: Conversation summary
        """
        if not history:
            return ""
        
        # Take most recent turns
        recent_turns = history[-max_turns:]
        
        summary_lines = ["Previous conversation:"]
        for i, turn in enumerate(recent_turns):
            summary_lines.append(f"User: {turn.query}")
            summary_lines.append(f"Assistant: {turn.response}")
        
        return "\n".join(summary_lines)
