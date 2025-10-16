"""
LLM lifecycle manager with telemetry, caching, and state management
Orchestrates all LLM operations with comprehensive monitoring
"""

import logging
import hashlib
import time
import threading
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from collections import OrderedDict

from .models import (
    LLMRequest, LLMResponse, LLMError, TelemetryData, HealthStatus,
    LLMState, ErrorCode, ErrorSeverity, ModelType, LLMConfig, CacheEntry
)
from .connection import OllamaConnectionManager
from .auth import ModelManager

logger = logging.getLogger(__name__)

class LLMManager:
    """
    Main LLM manager that orchestrates connections, models, and operations
    with comprehensive telemetry and caching.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.state = LLMState.DISCONNECTED
        
        # Initialize components
        self.connection = OllamaConnectionManager(config)
        self.model_manager = ModelManager(self.connection, config)
        
        # Cache setup
        self.response_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Telemetry and state
        self.telemetry_data: List[TelemetryData] = []
        self.request_counter = 0
        self.health_check_counter = 0
        
        # Threading
        self._lock = threading.RLock()
        self._shutdown = False
        
        # Performance metrics
        self.total_processing_time = 0.0
        self.total_tokens_processed = 0
        
        logger.info(f"LLM Manager initialized for model: {config.model.value}")
    
    def initialize(self) -> bool:
        """
        Initialize the LLM system - connection, model verification, etc.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        logger.info("Initializing LLM system...")
        self.state = LLMState.CONNECTING
        
        try:
            # Step 1: Verify environment
            if not self.model_manager.verify_environment():
                logger.error("Ollama environment verification failed")
                self.state = LLMState.ERROR
                return False
            
            # Step 2: Check if model is available
            model_health = self.model_manager.verify_model(self.config.model)
            
            if model_health.state == LLMState.ERROR and not self.model_manager.is_model_available(self.config.model):
                logger.info(f"Model {self.config.model.value} not found, attempting download...")
                if not self.model_manager.download_model(self.config.model):
                    logger.error(f"Failed to download model: {self.config.model.value}")
                    self.state = LLMState.ERROR
                    return False
                # Re-verify after download
                model_health = self.model_manager.verify_model(self.config.model)
            
            if model_health.state in [LLMState.READY, LLMState.DEGRADED]:
                self.state = model_health.state
                logger.info(f"LLM system initialized successfully. State: {self.state.value}")
                return True
            else:
                logger.error(f"Model verification failed. State: {model_health.state}")
                self.state = LLMState.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.state = LLMState.ERROR
            return False
    
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response for the given request with full telemetry
        
        Args:
            request: The LLM request
            
        Returns:
            LLMResponse: The generated response
            
        Raises:
            LLMError: If generation fails
        """
        start_time = time.time()
        request_id = request.request_id or self._generate_request_id()
        telemetry = TelemetryData(
            request_id=request_id,
            model=self.config.model,
            start_time=datetime.now(),
            end_time=datetime.now(),
            processing_time=0.0,
            tokens_input=0,
            tokens_output=0,
            cpu_time=0.0,
            memory_used=0,
            state=self.state,
            cache_hit=False
        )
        
        try:
            # Check cache first
            cached_response = self._get_cached_response(request)
            if cached_response:
                self.cache_hits += 1
                telemetry.cache_hit = True
                telemetry.end_time = datetime.now()
                telemetry.processing_time = time.time() - start_time
                self._record_telemetry(telemetry)
                
                logger.debug(f"Cache hit for request {request_id}")
                return cached_response
            
            self.cache_misses += 1
            
            # Ensure system is ready
            if not self._ensure_ready():
                raise LLMError(
                    code=ErrorCode.MODEL_LOAD_FAILED,
                    message="LLM system is not ready",
                    severity=ErrorSeverity.HIGH,
                    request_id=request_id
                )
            
            # Perform health check if needed
            self._perform_health_check_if_needed()
            
            # Generate response
            response_data = self._call_ollama_api(request, request_id)
            
            # Create response object
            processing_time = time.time() - start_time
            response = LLMResponse(
                content=response_data["response"],
                request_id=request_id,
                model_used=self.config.model,
                tokens_used=response_data.get("total_duration", 0),
                processing_time=processing_time,
                timestamp=datetime.now(),
                context_used=request.context
            )
            
            # Update telemetry
            telemetry.end_time = datetime.now()
            telemetry.processing_time = processing_time
            telemetry.tokens_output = len(response.content.split())  # Approximate
            telemetry.tokens_input = len(request.prompt.split())  # Approximate
            self._record_telemetry(telemetry)
            
            # Cache the response
            self._cache_response(request, response)
            
            # Update performance metrics
            self.total_processing_time += processing_time
            self.total_tokens_processed += telemetry.tokens_input + telemetry.tokens_output
            
            logger.info(f"Request {request_id} completed in {processing_time:.2f}s")
            return response
            
        except LLMError:
            # Re-raise LLM errors
            telemetry.error = LLMError(
                code=ErrorCode.REQUEST_TIMEOUT,
                message="Request failed",
                severity=ErrorSeverity.HIGH,
                request_id=request_id
            )
            telemetry.end_time = datetime.now()
            telemetry.processing_time = time.time() - start_time
            self._record_telemetry(telemetry)
            raise
            
        except Exception as e:
            # Handle unexpected errors
            error = LLMError(
                code=ErrorCode.INVALID_REQUEST,
                message=f"Unexpected error: {str(e)}",
                severity=ErrorSeverity.HIGH,
                request_id=request_id,
                details={"exception_type": type(e).__name__}
            )
            telemetry.error = error
            telemetry.end_time = datetime.now()
            telemetry.processing_time = time.time() - start_time
            self._record_telemetry(telemetry)
            
            logger.error(f"Unexpected error in request {request_id}: {e}")
            raise error
    
    def _call_ollama_api(self, request: LLMRequest, request_id: str) -> Dict[str, Any]:
        """
        Make actual API call to Ollama
        
        Args:
            request: The LLM request
            request_id: Request identifier
            
        Returns:
            Dict with response data
            
        Raises:
            LLMError: If API call fails
        """
        model_name = self.model_manager.MODEL_SPECS[self.config.model]["name"]
        
        request_data = {
            "model": model_name,
            "prompt": request.prompt,
            "stream": request.stream,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "top_p": self.config.top_p
            }
        }
        
        # Add context if provided (for RAG)
        if request.context:
            enhanced_prompt = f"Context: {request.context}\n\nQuestion: {request.prompt}\n\nAnswer:"
            request_data["prompt"] = enhanced_prompt
        
        try:
            response = self.connection.post("/api/generate", request_data)
            
            if response.status_code == 200:
                return response.json()
            else:
                raise LLMError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"Ollama API returned status {response.status_code}",
                    severity=ErrorSeverity.HIGH,
                    request_id=request_id,
                    details={"http_status": response.status_code}
                )
                
        except LLMError:
            raise  # Re-raise connection errors
        except Exception as e:
            raise LLMError(
                code=ErrorCode.MALFORMED_RESPONSE,
                message=f"Failed to parse Ollama response: {str(e)}",
                severity=ErrorSeverity.HIGH,
                request_id=request_id
            )
    
    def _ensure_ready(self) -> bool:
        """
        Ensure the LLM system is ready to process requests
        
        Returns:
            bool: True if ready, False otherwise
        """
        if self.state in [LLMState.READY, LLMState.DEGRADED]:
            return True
        
        if self.state == LLMState.ERROR:
            logger.info("Attempting to recover from error state...")
            return self.initialize()
        
        return self.initialize()
    
    def _perform_health_check_if_needed(self):
        """Perform health check based on configured frequency"""
        self.health_check_counter += 1
        
        if self.health_check_counter >= self.config.health_check_frequency:
            self.health_check_counter = 0
            health_status = self.model_manager.verify_model(self.config.model)
            self.state = health_status.state
            
            if health_status.state == LLMState.ERROR:
                logger.warning("Health check failed, attempting recovery...")
                self.initialize()
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        self.request_counter += 1
        timestamp = int(time.time() * 1000)
        return f"req_{timestamp}_{self.request_counter}"
    
    def _get_cached_response(self, request: LLMRequest) -> Optional[LLMResponse]:
        """
        Get cached response for request if available
        
        Args:
            request: The LLM request
            
        Returns:
            Cached response or None
        """
        if not self.config.cache_ttl:
            return None
        
        cache_key = self._generate_cache_key(request)
        
        with self._lock:
            if cache_key in self.response_cache:
                entry = self.response_cache[cache_key]
                
                # Check if cache entry is still valid
                if datetime.now() - entry.created_at < timedelta(seconds=self.config.cache_ttl):
                    # Update access info and move to end (LRU)
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    self.response_cache.move_to_end(cache_key)
                    return entry.response
                else:
                    # Remove expired entry
                    del self.response_cache[cache_key]
        
        return None
    
    def _cache_response(self, request: LLMRequest, response: LLMResponse):
        """
        Cache the response for future use
        
        Args:
            request: The LLM request
            response: The response to cache
        """
        if not self.config.cache_ttl or not self.config.cache_max_size:
            return
        
        cache_key = self._generate_cache_key(request)
        
        with self._lock:
            # Remove if already exists
            if cache_key in self.response_cache:
                del self.response_cache[cache_key]
            
            # Add new entry
            entry = CacheEntry(
                prompt_hash=cache_key,
                response=response,
                created_at=datetime.now(),
                access_count=1,
                last_accessed=datetime.now()
            )
            
            self.response_cache[cache_key] = entry
            
            # Enforce cache size limit
            if len(self.response_cache) > self.config.cache_max_size:
                # Remove oldest entry (first in OrderedDict)
                self.response_cache.popitem(last=False)
    
    def _generate_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key from request parameters"""
        content = f"{request.prompt}:{request.temperature}:{request.max_tokens}:{request.context or ''}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _record_telemetry(self, telemetry: TelemetryData):
        """Record telemetry data"""
        if self.config.enable_telemetry:
            self.telemetry_data.append(telemetry)
            # Keep only last 1000 telemetry records
            if len(self.telemetry_data) > 1000:
                self.telemetry_data.pop(0)
    
    def get_health_status(self) -> HealthStatus:
        """Get current health status of the LLM system"""
        health = self.model_manager.verify_model(self.config.model)
        self.state = health.state
        return health
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics and statistics"""
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        avg_processing_time = (
            self.total_processing_time / self.cache_misses 
            if self.cache_misses > 0 else 0
        )
        
        return {
            "state": self.state.value,
            "total_requests": total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{cache_hit_rate:.1%}",
            "avg_processing_time": f"{avg_processing_time:.2f}s",
            "total_tokens_processed": self.total_tokens_processed,
            "telemetry_records": len(self.telemetry_data),
            "cache_size": len(self.response_cache),
            "model": self.config.model.value
        }
    
    def clear_cache(self):
        """Clear the response cache"""
        with self._lock:
            self.response_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
        logger.info("Response cache cleared")
    
    def shutdown(self):
        """Gracefully shutdown the LLM manager"""
        self._shutdown = True
        self.connection.close()
        self.state = LLMState.DISCONNECTED
        logger.info("LLM Manager shutdown complete")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
