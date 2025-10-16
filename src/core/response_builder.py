"""
Response builder for RAG system
Integrates LLM with retrieved context to generate answers with citations
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import time

from .models import (
    RAGQuery, RAGResponse, RetrievedContext, DocumentChunk,
    ResponseQuality, ConversationTurn
)
from llm import LLMManager, LLMRequest, ModelType, LLMResponse

logger = logging.getLogger(__name__)

class ResponseBuilder:
    """
    Builds RAG responses by integrating LLM with retrieved context
    Handles citation generation and response formatting
    """
    
    def __init__(self, llm_manager: LLMManager, enable_citations: bool = True):
        self.llm_manager = llm_manager
        self.enable_citations = enable_citations
        
        # Citation patterns for extracting source references
        self.citation_patterns = [
            r'\[(\d+)\]',  # [1], [2], etc.
            r'\(source\s*:\s*(\w+)\)',  # (source: doc1)
            r'\[source:\s*([^\]]+)\]',  # [source: document1]
        ]
        self.compiled_citation_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.citation_patterns]
        
        logger.info("Response Builder initialized")
    
    def build_response(self, 
                      query: RAGQuery, 
                      context: RetrievedContext,
                      conversation_history: List[ConversationTurn] = None) -> RAGResponse:
        """
        Build a RAG response using retrieved context and LLM
        
        Args:
            query: Processed RAG query
            context: Retrieved document chunks
            conversation_history: Previous conversation turns
            
        Returns:
            RAGResponse: Generated response with citations
        """
        start_time = time.time()
        logger.info(f"Building response for query: '{query.original_query}'")
        
        try:
            # Prepare context for LLM
            context_text = self._prepare_context_for_llm(context.chunks)
            
            # Build conversation context
            conversation_context = self._build_conversation_context(conversation_history)
            
            # Generate LLM prompt
            prompt = self._build_llm_prompt(
                query.original_query, 
                context_text, 
                conversation_context,
                query.is_follow_up
            )
            
            # Generate response using LLM
            llm_response = self._generate_llm_response(prompt, query)
            
            # Extract citations from response
            citations = self._extract_citations(llm_response.content, context.chunks)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(context, llm_response, citations)
            
            # Determine response quality
            quality = self._assess_response_quality(confidence_score, context, citations)
            
            # Create RAG response
            processing_time = time.time() - start_time
            response = RAGResponse(
                answer=llm_response.content,
                original_query=query.original_query,
                contexts_used=context.chunks,
                citations=citations,
                quality=quality,
                confidence_score=confidence_score,
                processing_time=processing_time,
                retrieval_metrics={
                    "retrieval_strategy": context.retrieval_strategy.value,
                    "search_type": context.search_type.value,
                    "chunks_used": len(context.chunks),
                    "average_similarity": context.average_score,
                    "llm_processing_time": llm_response.processing_time
                }
            )
            
            logger.info(f"Response built in {processing_time:.2f}s with {len(citations)} citations")
            return response
            
        except Exception as e:
            logger.error(f"Response building failed: {e}")
            # Return error response
            return self._create_error_response(query.original_query, str(e))
    
    def _prepare_context_for_llm(self, chunks: List[DocumentChunk]) -> str:
        """
        Prepare retrieved chunks as context for LLM
        
        Args:
            chunks: Retrieved document chunks
            
        Returns:
            str: Formatted context text
        """
        if not chunks:
            return "No relevant information found in the documents."
        
        context_lines = ["Relevant information from documents:"]
        
        for i, chunk in enumerate(chunks, 1):
            source_info = f"Source {i}:"
            if chunk.document_id:
                source_info += f" {chunk.document_id}"
            if chunk.section_title:
                source_info += f" - {chunk.section_title}"
            if chunk.page_number:
                source_info += f" (Page {chunk.page_number})"
            
            context_lines.append(source_info)
            context_lines.append(f"Content: {chunk.content}")
            context_lines.append("")  # Empty line between chunks
        
        return "\n".join(context_lines)
    
    def _build_conversation_context(self, conversation_history: List[ConversationTurn]) -> str:
        """
        Build conversation context from history
        
        Args:
            conversation_history: Previous conversation turns
            
        Returns:
            str: Formatted conversation context
        """
        if not conversation_history:
            return ""
        
        # Take only recent turns to avoid context overflow
        recent_turns = conversation_history[-3:]  # Last 3 turns
        
        context_lines = ["Previous conversation:"]
        for turn in recent_turns:
            context_lines.append(f"User: {turn.query}")
            context_lines.append(f"Assistant: {turn.response}")
        
        return "\n".join(context_lines)
    
    def _build_llm_prompt(self, 
                         query: str, 
                         context: str, 
                         conversation_context: str,
                         is_follow_up: bool) -> str:
        """
        Build LLM prompt with context and instructions
        
        Args:
            query: User question
            context: Retrieved document context
            conversation_context: Previous conversation
            is_follow_up: Whether this is a follow-up question
            
        Returns:
            str: Complete LLM prompt
        """
        base_instructions = """
        You are a helpful assistant that answers questions based on the provided context.
        
        Instructions:
        1. Answer the question using ONLY the information from the provided context
        2. If the context doesn't contain relevant information, say so clearly
        3. Be concise and factual - avoid unnecessary elaboration
        4. Cite your sources using the format [Source X] where X is the source number
        5. If you're not sure, indicate the uncertainty
        """
        
        if is_follow_up:
            base_instructions += """
        6. Consider the conversation history when answering follow-up questions
        7. Maintain consistency with previous responses
            """
        
        prompt_parts = [base_instructions.strip()]
        
        if conversation_context:
            prompt_parts.append(conversation_context)
        
        prompt_parts.append(context)
        prompt_parts.append(f"Question: {query}")
        prompt_parts.append("Answer:")
        
        return "\n\n".join(prompt_parts)
    
    def _generate_llm_response(self, prompt: str, query: RAGQuery) -> LLMResponse:
        """
        Generate response using LLM
        
        Args:
            prompt: Complete LLM prompt
            query: RAG query for context
            
        Returns:
            LLMResponse: LLM-generated response
        """
        llm_request = LLMRequest(
            prompt=prompt,
            model=ModelType.TINYLLAMA,
            temperature=0.1,  # Low temperature for factual responses
            max_tokens=500,
            context=query.original_query,
            request_id=f"rag_{int(time.time())}"
        )
        
        try:
            return self.llm_manager.generate_response(llm_request)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Return a fallback response
            return LLMResponse(
                content="I apologize, but I encountered an error while generating a response. Please try again.",
                request_id=llm_request.request_id,
                model_used=ModelType.TINYLLAMA,
                tokens_used=0,
                processing_time=0.0,
                timestamp=datetime.now()
            )
    
    def _extract_citations(self, response_text: str, chunks: List[DocumentChunk]) -> Dict[str, List[str]]:
        """
        Extract citations from response text and map to document chunks
        
        Args:
            response_text: LLM-generated response
            chunks: Retrieved document chunks
            
        Returns:
            Dict mapping citation IDs to response segments
        """
        if not self.enable_citations or not chunks:
            return {}
        
        citations = {}
        
        # Look for citation patterns in the response
        for pattern in self.compiled_citation_patterns:
            matches = pattern.findall(response_text)
            for match in matches:
                # Try to map the citation to a chunk
                citation_id = self._map_citation_to_chunk(match, chunks)
                if citation_id:
                    # Find the text segment around this citation
                    segment = self._extract_citation_segment(response_text, match, pattern)
                    if citation_id not in citations:
                        citations[citation_id] = []
                    citations[citation_id].append(segment)
        
        return citations
    
    def _map_citation_to_chunk(self, citation_ref: str, chunks: List[DocumentChunk]) -> Optional[str]:
        """
        Map a citation reference to a specific document chunk
        
        Args:
            citation_ref: Citation reference from response
            chunks: Retrieved document chunks
            
        Returns:
            Citation ID or None if no match found
        """
        # Simple mapping: numeric citations map to chunk order
        if citation_ref.isdigit():
            index = int(citation_ref) - 1  # Convert to 0-based index
            if 0 <= index < len(chunks):
                return chunks[index].citation_id
        
        # Try to match by document ID or other identifiers
        for chunk in chunks:
            if (citation_ref.lower() in chunk.document_id.lower() or 
                (chunk.section_title and citation_ref.lower() in chunk.section_title.lower())):
                return chunk.citation_id
        
        return None
    
    def _extract_citation_segment(self, response_text: str, citation: str, pattern: re.Pattern) -> str:
        """
        Extract the text segment around a citation
        
        Args:
            response_text: Full response text
            citation: Citation reference
            pattern: Regex pattern used to find the citation
            
        Returns:
            str: Text segment containing the citation
        """
        # Find the position of the citation
        match = pattern.search(response_text)
        if not match:
            return citation
        
        start_pos = max(0, match.start() - 50)  # 50 chars before
        end_pos = min(len(response_text), match.end() + 50)  # 50 chars after
        
        segment = response_text[start_pos:end_pos]
        
        # Clean up the segment
        if start_pos > 0 and not segment.startswith(' '):
            segment = '...' + segment
        if end_pos < len(response_text) and not segment.endswith(' '):
            segment = segment + '...'
        
        return segment.strip()
    
    def _calculate_confidence_score(self, 
                                  context: RetrievedContext, 
                                  llm_response: LLMResponse,
                                  citations: Dict[str, List[str]]) -> float:
        """
        Calculate confidence score for the response
        
        Args:
            context: Retrieved context
            llm_response: LLM response
            citations: Extracted citations
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        score = 0.0
        
        # Base score from retrieval quality
        if context.chunks:
            score += min(context.average_score, 1.0) * 0.4  # 40% weight
        
        # Score from citation density
        if citations:
            citation_density = len(citations) / len(context.chunks) if context.chunks else 0
            score += min(citation_density, 1.0) * 0.3  # 30% weight
        
        # Score from response quality indicators
        response_quality_score = self._assess_response_quality_indicators(llm_response.content)
        score += response_quality_score * 0.3  # 30% weight
        
        return min(score, 1.0)
    
    def _assess_response_quality_indicators(self, response_text: str) -> float:
        """
        Assess response quality based on text indicators
        
        Args:
            response_text: LLM response text
            
        Returns:
            float: Quality score between 0.0 and 1.0
        """
        score = 0.5  # Base score
        
        # Positive indicators
        positive_indicators = [
            r'according to',
            r'based on',
            r'the (document|source|context)',
            r'\[Source \d+\]',
            r'as mentioned in',
        ]
        
        for indicator in positive_indicators:
            if re.search(indicator, response_text, re.IGNORECASE):
                score += 0.1
        
        # Negative indicators (uncertainty, lack of information)
        negative_indicators = [
            r'i don\'t know',
            r'i cannot (answer|find)',
            r'no information',
            r'not (provided|available|found)',
            r'unable to',
        ]
        
        for indicator in negative_indicators:
            if re.search(indicator, response_text, re.IGNORECASE):
                score -= 0.1
        
        return max(0.0, min(score, 1.0))
    
    def _assess_response_quality(self, 
                               confidence_score: float, 
                               context: RetrievedContext,
                               citations: Dict[str, List[str]]) -> ResponseQuality:
        """
        Assess overall response quality
        
        Args:
            confidence_score: Calculated confidence score
            context: Retrieved context
            citations: Extracted citations
            
        Returns:
            ResponseQuality: Quality assessment
        """
        if confidence_score >= 0.8:
            return ResponseQuality.EXCELLENT
        elif confidence_score >= 0.6:
            return ResponseQuality.GOOD
        elif confidence_score >= 0.4:
            return ResponseQuality.FAIR
        else:
            return ResponseQuality.POOR
    
    def _create_error_response(self, query: str, error_message: str) -> RAGResponse:
        """
        Create an error response when response building fails
        
        Args:
            query: Original user query
            error_message: Error details
            
        Returns:
            RAGResponse: Error response
        """
        return RAGResponse(
            answer=f"I apologize, but I encountered an error while processing your question: '{query}'. Please try again or rephrase your question.",
            original_query=query,
            contexts_used=[],
            citations={},
            quality=ResponseQuality.POOR,
            confidence_score=0.0,
            processing_time=0.0,
            retrieval_metrics={"error": error_message}
        )
    
    def format_response_for_display(self, response: RAGResponse) -> str:
        """
        Format RAG response for display with citations
        
        Args:
            response: RAG response to format
            
        Returns:
            str: Formatted response text
        """
        return response.format_with_citations()
