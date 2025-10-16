"""
Output handler for LLM responses
Handles response validation, formatting, error handling, and quality checks
"""

import logging
import re
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from .models import (
    LLMResponse, LLMError, LLMRequest, TelemetryData,
    ErrorCode, ErrorSeverity, ModelType
)

logger = logging.getLogger(__name__)

class OutputHandler:
    """
    Handles LLM output processing, validation, formatting, and quality assurance
    """
    
    def __init__(self):
        # Quality thresholds
        self.min_response_length = 10  # characters
        self.max_response_length = 5000  # characters
        self.max_processing_time = 30.0  # seconds
        
        # Pattern for detecting problematic responses
        self.problematic_patterns = [
            r"^(sorry|i cannot|i don't|i'm unable|as an ai)",
            r"\b(cannot|can't|unable to|don't know|no information)\b",
            r"^(please provide|can you provide|give me more)",
            r"\b(error|failed|invalid|not supported)\b"
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.problematic_patterns]
        
        logger.info("Output Handler initialized")
    
    def process_response(self, raw_response: Dict[str, Any], request: LLMRequest, 
                        telemetry: TelemetryData) -> LLMResponse:
        """
        Process and validate raw LLM response
        
        Args:
            raw_response: Raw response from LLM API
            request: Original LLM request
            telemetry: Telemetry data for the request
            
        Returns:
            LLMResponse: Processed and validated response
            
        Raises:
            LLMError: If response validation fails
        """
        try:
            # Extract response content
            response_content = self._extract_response_content(raw_response)
            
            # Validate basic response
            self._validate_basic_response(response_content, telemetry)
            
            # Apply quality checks
            quality_issues = self._check_response_quality(response_content, request)
            
            # Format response
            formatted_response = self._format_response(response_content, request)
            
            # Create response object
            response = LLMResponse(
                content=formatted_response,
                request_id=request.request_id or "unknown",
                model_used=request.model,
                tokens_used=raw_response.get("total_duration", 0) or len(formatted_response.split()),
                processing_time=telemetry.processing_time,
                timestamp=datetime.now(),
                context_used=request.context
            )
            
            # Log quality issues as warnings
            if quality_issues:
                logger.warning(f"Quality issues in response {request.request_id}: {quality_issues}")
            
            logger.debug(f"Response processed successfully for request {request.request_id}")
            return response
            
        except LLMError:
            raise  # Re-raise validation errors
        except Exception as e:
            logger.error(f"Unexpected error processing response: {e}")
            raise LLMError(
                code=ErrorCode.MALFORMED_RESPONSE,
                message=f"Failed to process LLM response: {str(e)}",
                severity=ErrorSeverity.HIGH,
                request_id=request.request_id,
                details={"exception_type": type(e).__name__}
            )
    
    def _extract_response_content(self, raw_response: Dict[str, Any]) -> str:
        """
        Extract response content from raw API response
        
        Args:
            raw_response: Raw response from LLM API
            
        Returns:
            str: Extracted response content
            
        Raises:
            LLMError: If content extraction fails
        """
        try:
            # Ollama API response structure
            if "response" in raw_response:
                content = raw_response["response"]
            elif "content" in raw_response:
                content = raw_response["content"]
            else:
                # Try to find any string field that might contain the response
                for key, value in raw_response.items():
                    if isinstance(value, str) and len(value) > 10:  # Heuristic for content
                        content = value
                        break
                else:
                    raise ValueError("No response content found in API response")
            
            if not isinstance(content, str):
                raise ValueError(f"Response content is not a string: {type(content)}")
            
            return content.strip()
            
        except Exception as e:
            raise LLMError(
                code=ErrorCode.MALFORMED_RESPONSE,
                message="Failed to extract response content from API response",
                severity=ErrorSeverity.HIGH,
                details={"raw_response_keys": list(raw_response.keys()), "error": str(e)}
            )
    
    def _validate_basic_response(self, response_content: str, telemetry: TelemetryData):
        """
        Perform basic validation on the response
        
        Args:
            response_content: The response content to validate
            telemetry: Telemetry data for context
            
        Raises:
            LLMError: If validation fails
        """
        # Check for empty response
        if not response_content:
            raise LLMError(
                code=ErrorCode.EMPTY_RESPONSE,
                message="LLM returned an empty response",
                severity=ErrorSeverity.HIGH,
                request_id=telemetry.request_id,
                details={"processing_time": telemetry.processing_time}
            )
        
        # Check response length
        if len(response_content) < self.min_response_length:
            raise LLMError(
                code=ErrorCode.QUALITY_ISSUE,
                message=f"Response too short ({len(response_content)} characters)",
                severity=ErrorSeverity.MEDIUM,
                request_id=telemetry.request_id,
                details={
                    "response_length": len(response_content),
                    "min_required": self.min_response_length,
                    "response_preview": response_content[:100] + "..." if len(response_content) > 100 else response_content
                }
            )
        
        if len(response_content) > self.max_response_length:
            logger.warning(f"Response very long ({len(response_content)} characters) for request {telemetry.request_id}")
        
        # Check processing time
        if telemetry.processing_time > self.max_processing_time:
            logger.warning(f"Slow response processing: {telemetry.processing_time:.2f}s for request {telemetry.request_id}")
    
    def _check_response_quality(self, response_content: str, request: LLMRequest) -> List[str]:
        """
        Check response quality and identify potential issues
        
        Args:
            response_content: The response content to check
            request: Original request for context
            
        Returns:
            List of quality issues found
        """
        issues = []
        
        # Check for problematic patterns
        for pattern in self.compiled_patterns:
            if pattern.search(response_content):
                issues.append(f"Matches problematic pattern: {pattern.pattern}")
                break
        
        # Check for repetition
        if self._has_excessive_repetition(response_content):
            issues.append("Excessive repetition in response")
        
        # Check for relevance (basic heuristic)
        if request.context and not self._check_relevance(response_content, request.context):
            issues.append("Possible relevance issue with provided context")
        
        # Check for completeness
        if self._is_incomplete_response(response_content):
            issues.append("Response appears incomplete")
        
        return issues
    
    def _has_excessive_repetition(self, text: str, threshold: float = 0.3) -> bool:
        """
        Check if text has excessive repetition
        
        Args:
            text: Text to check
            threshold: Repetition threshold (0.0 to 1.0)
            
        Returns:
            bool: True if excessive repetition detected
        """
        words = text.lower().split()
        if len(words) < 10:  # Too short to determine repetition
            return False
        
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Calculate repetition ratio
        max_count = max(word_counts.values())
        repetition_ratio = max_count / len(words)
        
        return repetition_ratio > threshold
    
    def _check_relevance(self, response: str, context: str) -> bool:
        """
        Basic relevance check between response and context
        
        Args:
            response: LLM response
            context: Provided context
            
        Returns:
            bool: True if response appears relevant to context
        """
        # Simple keyword overlap check
        context_words = set(context.lower().split())
        response_words = set(response.lower().split())
        
        # Consider relevant if at least 2 context words appear in response
        overlap = context_words.intersection(response_words)
        return len(overlap) >= 2
    
    def _is_incomplete_response(self, text: str) -> bool:
        """
        Check if response appears incomplete
        
        Args:
            text: Response text
            
        Returns:
            bool: True if response appears incomplete
        """
        # Check for trailing ellipsis or cut-off sentences
        if text.strip().endswith(('...', '…', '--', '---')):
            return True
        
        # Check if last sentence is incomplete
        sentences = re.split(r'[.!?]', text)
        if sentences and sentences[-1].strip() and not text.rstrip().endswith(('.', '!', '?')):
            return True
        
        return False
    
    def _format_response(self, response_content: str, request: LLMRequest) -> str:
        """
        Format the response based on request type and context
        
        Args:
            response_content: Raw response content
            request: Original request
            
        Returns:
            str: Formatted response
        """
        formatted = response_content.strip()
        
        # Remove any leading/trailing quotes if they don't make sense
        if formatted.startswith('"') and formatted.endswith('"'):
            if len(formatted) > 2 and not formatted[1:-1].count('"') > 1:
                formatted = formatted[1:-1]
        
        # Ensure proper capitalization for the first letter
        if formatted and formatted[0].islower():
            formatted = formatted[0].upper() + formatted[1:]
        
        # Remove excessive whitespace
        formatted = re.sub(r'\n\s*\n', '\n\n', formatted)  # Multiple newlines to double
        formatted = re.sub(r'[ \t]+', ' ', formatted)  # Multiple spaces to single
        
        return formatted
    
    def create_error_response(self, error: LLMError, request: LLMRequest) -> LLMResponse:
        """
        Create a user-friendly error response
        
        Args:
            error: The LLM error
            request: Original request
            
        Returns:
            LLMResponse: Error response with helpful message
        """
        # User-friendly error messages based on error type
        error_messages = {
            ErrorCode.OLLAMA_NOT_RUNNING: (
                "I cannot connect to the AI engine. "
                "Please make sure Ollama is installed and running on your system."
            ),
            ErrorCode.CONNECTION_TIMEOUT: (
                "The request took too long to process. "
                "This might be due to high load or network issues. Please try again."
            ),
            ErrorCode.MODEL_NOT_FOUND: (
                "The AI model is not available. "
                "Please check if the model is installed in Ollama."
            ),
            ErrorCode.REQUEST_TIMEOUT: (
                "The request timed out. "
                "This might be due to the complexity of your question or system load."
            ),
            ErrorCode.EMPTY_RESPONSE: (
                "The AI returned an empty response. "
                "This might be due to the question format or model limitations."
            ),
            ErrorCode.QUALITY_ISSUE: (
                "The response didn't meet quality standards. "
                "Please try rephrasing your question or providing more context."
            )
        }
        
        # Default message for unhandled error types
        user_message = error_messages.get(
            error.code,
            "An unexpected error occurred while processing your request. Please try again."
        )
        
        # Add developer details for high severity errors
        if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            user_message += f"\n\n[Technical details: {error.code.value}]"
        
        return LLMResponse(
            content=user_message,
            request_id=request.request_id or "unknown",
            model_used=request.model,
            tokens_used=0,
            processing_time=0.0,
            timestamp=datetime.now(),
            context_used=request.context
        )
    
    def get_quality_metrics(self, response: LLMResponse) -> Dict[str, Any]:
        """
        Calculate quality metrics for a response
        
        Args:
            response: The LLM response
            
        Returns:
            Dict with quality metrics
        """
        content = response.content
        
        metrics = {
            "response_length": len(content),
            "word_count": len(content.split()),
            "sentence_count": len(re.split(r'[.!?]', content)),
            "avg_word_length": sum(len(word) for word in content.split()) / max(len(content.split()), 1),
            "has_quality_issues": bool(self._check_response_quality(content, LLMRequest(prompt=""))),
            "processing_time": response.processing_time,
            "tokens_per_second": response.tokens_used / response.processing_time if response.processing_time > 0 else 0
        }
        
        return metrics
