"""
Quality checker for RAG system
Validates responses, detects hallucinations, and assesses answer quality
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from .models import (
    RAGResponse, RAGQuery, DocumentChunk, QualityMetrics,
    ResponseQuality, RetrievedContext
)
from llm import LLMManager, LLMRequest, ModelType

logger = logging.getLogger(__name__)

class QualityChecker:
    """
    Validates RAG responses for quality, relevance, and factual accuracy
    Detects hallucinations and assesses overall response quality
    """
    
    def __init__(self, llm_manager: LLMManager, enable_llm_validation: bool = True):
        self.llm_manager = llm_manager
        self.enable_llm_validation = enable_llm_validation
        
        # Hallucination detection patterns
        self.hallucination_indicators = [
            r'\b(studies show|research indicates|experts say)\b',
            r'\b(according to studies|based on research)\b',
            r'\b(it is well known|it is common knowledge)\b',
            r'\b(many people|most experts)\b',
            r'\b(always|never|everyone|nobody)\b',  # Overgeneralizations
        ]
        self.compiled_indicators = [re.compile(pattern, re.IGNORECASE) for pattern in self.hallucination_indicators]
        
        # Quality assessment thresholds
        self.quality_thresholds = {
            ResponseQuality.EXCELLENT: 0.8,
            ResponseQuality.GOOD: 0.6,
            ResponseQuality.FAIR: 0.4,
            ResponseQuality.POOR: 0.0
        }
        
        logger.info("Quality Checker initialized")
    
    def validate_response(self, 
                         response: RAGResponse, 
                         query: RAGQuery,
                         context: RetrievedContext) -> QualityMetrics:
        """
        Comprehensive validation of RAG response quality
        
        Args:
            response: RAG response to validate
            query: Original query
            context: Retrieved context used for response
            
        Returns:
            QualityMetrics: Comprehensive quality assessment
        """
        logger.info(f"Validating response quality for query: '{query.original_query}'")
        
        try:
            # Calculate various quality metrics
            answer_relevance = self._calculate_answer_relevance(response.answer, query.original_query)
            context_relevance = self._calculate_context_relevance(response.answer, context.chunks)
            citation_density = self._calculate_citation_density(response.answer, response.citations)
            hallucination_score = self._detect_hallucinations(response.answer, context.chunks)
            coherence_score = self._assess_coherence(response.answer)
            
            # Optional: LLM-based validation for more sophisticated checking
            if self.enable_llm_validation:
                llm_validation = self._llm_quality_validation(response, query, context)
                # Blend LLM validation with automated metrics
                answer_relevance = (answer_relevance + llm_validation.get("relevance", 0.5)) / 2
                hallucination_score = (hallucination_score + llm_validation.get("hallucination", 0.5)) / 2
            
            metrics = QualityMetrics(
                answer_relevance=answer_relevance,
                context_relevance=context_relevance,
                citation_density=citation_density,
                hallucination_score=hallucination_score,
                coherence_score=coherence_score
            )
            
            logger.info(f"Quality validation completed: relevance={answer_relevance:.2f}, "
                       f"hallucination={hallucination_score:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            # Return default metrics on failure
            return QualityMetrics(
                answer_relevance=0.5,
                context_relevance=0.5,
                citation_density=0.0,
                hallucination_score=0.5,
                coherence_score=0.5
            )
    
    def _calculate_answer_relevance(self, answer: str, query: str) -> float:
        """
        Calculate how relevant the answer is to the original query
        
        Args:
            answer: Generated answer
            query: Original user query
            
        Returns:
            float: Relevance score between 0.0 and 1.0
        """
        if not answer or not query:
            return 0.0
        
        # Convert to lowercase for comparison
        answer_lower = answer.lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # Exact phrase matching
        if query_lower in answer_lower:
            score += 0.3
        
        # Keyword overlap
        query_words = set(query_lower.split())
        answer_words = set(answer_lower.split())
        overlap = query_words.intersection(answer_words)
        
        if query_words:
            keyword_score = len(overlap) / len(query_words)
            score += keyword_score * 0.4
        
        # Question answering pattern matching
        question_patterns = [
            r'(what|who|when|where|why|how)\s+is',
            r'explain',
            r'describe',
            r'tell me about'
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, query_lower) and len(answer.split()) > 10:
                score += 0.3  # Bonus for substantive answers to questions
                break
        
        return min(score, 1.0)
    
    def _calculate_context_relevance(self, answer: str, chunks: List[DocumentChunk]) -> float:
        """
        Calculate how well the answer is supported by the context
        
        Args:
            answer: Generated answer
            chunks: Retrieved document chunks
            
        Returns:
            float: Context relevance score between 0.0 and 1.0
        """
        if not answer or not chunks:
            return 0.0
        
        answer_lower = answer.lower()
        total_support = 0.0
        
        for chunk in chunks:
            chunk_lower = chunk.content.lower()
            
            # Calculate overlap between answer and chunk
            answer_words = set(answer_lower.split())
            chunk_words = set(chunk_lower.split())
            overlap = answer_words.intersection(chunk_words)
            
            if answer_words:
                chunk_support = len(overlap) / len(answer_words)
                total_support += chunk_support
        
        # Average support across all chunks
        avg_support = total_support / len(chunks) if chunks else 0.0
        
        # Penalize answers that are too long compared to context (potential hallucination)
        answer_length = len(answer.split())
        avg_chunk_length = sum(len(chunk.content.split()) for chunk in chunks) / len(chunks) if chunks else 0
        
        if avg_chunk_length > 0 and answer_length > avg_chunk_length * 3:
            avg_support *= 0.7  # Penalize overly long answers
        
        return min(avg_support, 1.0)
    
    def _calculate_citation_density(self, answer: str, citations: Dict[str, List[str]]) -> float:
        """
        Calculate citation density in the answer
        
        Args:
            answer: Generated answer
            citations: Citation mapping
            
        Returns:
            float: Citation density score between 0.0 and 1.0
        """
        if not answer or not citations:
            return 0.0
        
        # Count sentences in answer
        sentences = re.split(r'[.!?]+', answer)
        sentence_count = len([s for s in sentences if s.strip()])
        
        if sentence_count == 0:
            return 0.0
        
        # Count citation references
        citation_count = 0
        for pattern in self.compiled_citation_patterns:
            citation_count += len(pattern.findall(answer))
        
        # Calculate density (citations per sentence)
        density = citation_count / sentence_count
        
        # Normalize to 0-1 scale (assuming 0.5 citations per sentence is good)
        normalized_density = min(density / 0.5, 1.0)
        
        return normalized_density
    
    def _detect_hallucinations(self, answer: str, chunks: List[DocumentChunk]) -> float:
        """
        Detect potential hallucinations in the answer
        
        Args:
            answer: Generated answer
            chunks: Retrieved document chunks
            
        Returns:
            float: Hallucination score (lower is better)
        """
        if not answer or not chunks:
            return 0.5  # Neutral score for empty cases
        
        hallucination_indicators = 0
        total_indicators_checked = len(self.compiled_indicators)
        
        # Check for hallucination indicators
        for pattern in self.compiled_indicators:
            if pattern.search(answer):
                hallucination_indicators += 1
        
        # Check for unsupported claims
        unsupported_claims = self._detect_unsupported_claims(answer, chunks)
        hallucination_indicators += unsupported_claims
        
        # Calculate score (lower is better)
        hallucination_score = hallucination_indicators / (total_indicators_checked + 5)  # +5 for claim checks
        
        return min(hallucination_score, 1.0)
    
    def _detect_unsupported_claims(self, answer: str, chunks: List[DocumentChunk]) -> int:
        """
        Detect claims in the answer that aren't supported by context
        
        Args:
            answer: Generated answer
            chunks: Retrieved document chunks
            
        Returns:
            int: Number of unsupported claims detected
        """
        # Extract factual claims from answer (simplified approach)
        claims = self._extract_claims(answer)
        unsupported_count = 0
        
        # Combine all context for checking
        all_context = " ".join(chunk.content for chunk in chunks).lower()
        
        for claim in claims:
            if not self._is_claim_supported(claim, all_context):
                unsupported_count += 1
        
        return unsupported_count
    
    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract potential factual claims from text
        
        Args:
            text: Text to extract claims from
            
        Returns:
            List of potential claims
        """
        # Simple claim extraction based on sentence structure
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        claim_indicators = [
            r'^[A-Z][^.!?]*(is|are|was|were|has|have|had)',
            r'^[A-Z][^.!?]*(can|could|will|would|should)',
            r'^[A-Z][^.!?]*(\d+|many|most|some|all)',
        ]
        
        compiled_indicators = [re.compile(pattern, re.IGNORECASE) for pattern in claim_indicators]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) > 3:  # Reasonable length for a claim
                for pattern in compiled_indicators:
                    if pattern.match(sentence):
                        claims.append(sentence)
                        break
        
        return claims
    
    def _is_claim_supported(self, claim: str, context: str) -> bool:
        """
        Check if a claim is supported by the context
        
        Args:
            claim: Factual claim to check
            context: Combined context text
            
        Returns:
            bool: True if claim appears supported
        """
        claim_lower = claim.lower()
        context_lower = context.lower()
        
        # Simple keyword matching (could be enhanced with semantic similarity)
        claim_words = set(claim_lower.split())
        context_words = set(context_lower.split())
        
        overlap = claim_words.intersection(context_words)
        
        # Consider supported if at least 30% of claim words appear in context
        if len(claim_words) > 0 and len(overlap) / len(claim_words) >= 0.3:
            return True
        
        return False
    
    def _assess_coherence(self, answer: str) -> float:
        """
        Assess the coherence and readability of the answer
        
        Args:
            answer: Generated answer
            
        Returns:
            float: Coherence score between 0.0 and 1.0
        """
        if not answer:
            return 0.0
        
        score = 0.5  # Base score
        
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', answer)
        valid_sentences = [s for s in sentences if len(s.strip().split()) >= 3]
        
        if sentences:
            sentence_quality = len(valid_sentences) / len(sentences)
            score += sentence_quality * 0.3
        
        # Check for logical connectors
        connectors = ['however', 'therefore', 'additionally', 'furthermore', 'consequently']
        connector_count = sum(1 for connector in connectors if connector in answer.lower())
        
        if len(sentences) > 1:
            connector_density = connector_count / (len(sentences) - 1)
            score += min(connector_density * 0.2, 0.2)
        
        return min(score, 1.0)
    
    def _llm_quality_validation(self, 
                              response: RAGResponse, 
                              query: RAGQuery,
                              context: RetrievedContext) -> Dict[str, float]:
        """
        Use LLM for more sophisticated quality validation
        
        Args:
            response: RAG response to validate
            query: Original query
            context: Retrieved context
            
        Returns:
            Dict with LLM-based quality scores
        """
        if not self.enable_llm_validation:
            return {}
        
        try:
            prompt = self._build_validation_prompt(response, query, context)
            
            llm_request = LLMRequest(
                prompt=prompt,
                model=ModelType.TINYLLAMA,
                temperature=0.1,
                max_tokens=200
            )
            
            llm_response = self.llm_manager.generate_response(llm_request)
            return self._parse_validation_response(llm_response.content)
            
        except Exception as e:
            logger.warning(f"LLM quality validation failed: {e}")
            return {}
    
    def _build_validation_prompt(self, 
                               response: RAGResponse, 
                               query: RAGQuery,
                               context: RetrievedContext) -> str:
        """
        Build prompt for LLM-based quality validation
        
        Args:
            response: RAG response
            query: Original query
            context: Retrieved context
            
        Returns:
            str: Validation prompt
        """
        context_preview = "\n".join([chunk.content[:200] + "..." for chunk in context.chunks[:3]])
        
        prompt = f"""
        Evaluate the quality of this AI response based on the question and context.
        
        QUESTION: {query.original_query}
        
        CONTEXT:
        {context_preview}
        
        RESPONSE:
        {response.answer}
        
        Please evaluate on these dimensions (0.0 to 1.0):
        1. Relevance: How well does the answer address the question?
        2. Support: How well is the answer supported by the context?
        3. Hallucination: How much unsupported information does it contain?
        
        Return your evaluation as: relevance=X.X,support=X.X,hallucination=X.X
        """
        
        return prompt
    
    def _parse_validation_response(self, response: str) -> Dict[str, float]:
        """
        Parse LLM validation response into scores
        
        Args:
            response: LLM validation response
            
        Returns:
            Dict with parsed scores
        """
        scores = {}
        
        try:
            # Look for score patterns
            patterns = {
                'relevance': r'relevance=([0-9.]+)',
                'support': r'support=([0-9.]+)',
                'hallucination': r'hallucination=([0-9.]+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, response)
                if match:
                    scores[key] = float(match.group(1))
        
        except Exception as e:
            logger.warning(f"Failed to parse validation response: {e}")
        
        return scores
    
    def should_show_response(self, response: RAGResponse, metrics: QualityMetrics, 
                           min_confidence: float = 0.6) -> Tuple[bool, str]:
        """
        Determine if response should be shown to user based on quality
        
        Args:
            response: RAG response
            metrics: Quality metrics
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (should_show, reason)
        """
        # Check confidence threshold
        if response.confidence_score < min_confidence:
            return False, f"Low confidence score: {response.confidence_score:.2f}"
        
        # Check for high hallucination
        if metrics.hallucination_score > 0.7:
            return False, f"High hallucination risk: {metrics.hallucination_score:.2f}"
        
        # Check for poor relevance
        if metrics.answer_relevance < 0.3:
            return False, f"Poor relevance to question: {metrics.answer_relevance:.2f}"
        
        # Check for very low context support
        if metrics.context_relevance < 0.2:
            return False, f"Insufficient context support: {metrics.context_relevance:.2f}"
        
        return True, "Quality standards met"
    
    def get_quality_feedback(self, response: RAGResponse, metrics: QualityMetrics) -> str:
        """
        Generate user-friendly quality feedback
        
        Args:
            response: RAG response
            metrics: Quality metrics
            
        Returns:
            str: User-friendly feedback
        """
        if response.quality == ResponseQuality.POOR:
            return "⚠️ The answer may not fully address your question or could contain unsupported information. Please consider verifying with additional sources."
        
        elif response.quality == ResponseQuality.FAIR:
            return "ℹ️ This answer is somewhat relevant but may lack comprehensive support from the provided documents."
        
        elif response.quality == ResponseQuality.GOOD:
            return "✅ This answer appears relevant and reasonably supported by the available information."
        
        else:  # EXCELLENT
            return "✅ This answer is highly relevant and well-supported by the provided documents."
