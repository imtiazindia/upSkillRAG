"""
Persistent Ollama connection manager
Handles connection pooling, health checks, and reconnection logic
"""

import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import LLMState, LLMConfig, HealthStatus, LLMError, ErrorCode, ErrorSeverity

# Configure logging
logger = logging.getLogger(__name__)

class OllamaConnectionManager:
    """
    Manages persistent connection to Ollama server with health monitoring
    and automatic reconnection capabilities.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.session: Optional[requests.Session] = None
        self.state = LLMState.DISCONNECTED
        self.last_health_check: Optional[datetime] = None
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        self.health_check_interval = timedelta(seconds=30)
        
        # Telemetry data
        self.response_times: list[float] = []
        self.error_count = 0
        self.request_count = 0
        
        self._setup_session()
    
    def _setup_session(self) -> None:
        """Configure HTTP session with retry strategy and connection pooling"""
        try:
            self.session = requests.Session()
            
            # Retry strategy for transient failures
            retry_strategy = Retry(
                total=self.config.max_retries,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE"],
                backoff_factor=1
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            
            logger.info("HTTP session configured with connection pooling and retry strategy")
            
        except Exception as e:
            logger.error(f"Failed to setup HTTP session: {e}")
            self.state = LLMState.ERROR
            raise
    
    def connect(self) -> bool:
        """
        Establish connection to Ollama server and verify availability
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self.state == LLMState.READY:
            logger.debug("Already connected to Ollama")
            return True
            
        self.state = LLMState.CONNECTING
        logger.info(f"Attempting to connect to Ollama at {self.config.base_url}")
        
        try:
            # Test basic connectivity
            response = self.session.get(f"{self.config.base_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                self.state = LLMState.READY
                self.connection_attempts = 0
                self.last_health_check = datetime.now()
                logger.info("Successfully connected to Ollama server")
                return True
            else:
                self._handle_connection_error(f"Ollama returned status code: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError as e:
            self._handle_connection_error(f"Cannot connect to Ollama: {e}")
            return False
            
        except requests.exceptions.Timeout as e:
            self._handle_connection_error(f"Connection timeout: {e}")
            return False
            
        except Exception as e:
            self._handle_connection_error(f"Unexpected connection error: {e}")
            return False
    
    def _handle_connection_error(self, error_message: str) -> None:
        """Handle connection errors with exponential backoff"""
        self.connection_attempts += 1
        self.state = LLMState.ERROR
        
        if self.connection_attempts >= self.max_connection_attempts:
            logger.critical(f"Max connection attempts reached: {error_message}")
        else:
            backoff_time = min(2 ** self.connection_attempts, 60)  # Exponential backoff, max 60s
            logger.warning(f"Connection attempt {self.connection_attempts} failed: {error_message}. Retrying in {backoff_time}s")
            time.sleep(backoff_time)
    
    def check_health(self) -> HealthStatus:
        """
        Perform health check on Ollama connection
        
        Returns:
            HealthStatus: Current health status of the connection
        """
        self.last_health_check = datetime.now()
        
        try:
            start_time = time.time()
            response = self.session.get(f"{self.config.base_url}/api/tags", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                # Update telemetry
                self.response_times.append(response_time)
                if len(self.response_times) > 100:  # Keep last 100 measurements
                    self.response_times.pop(0)
                
                avg_response_time = sum(self.response_times) / len(self.response_times)
                error_rate = self.error_count / max(self.request_count, 1)
                
                # Determine state based on performance
                if response_time > 5.0:  # Slow response threshold
                    self.state = LLMState.DEGRADED
                else:
                    self.state = LLMState.READY
                
                return HealthStatus(
                    state=self.state,
                    last_health_check=self.last_health_check,
                    avg_response_time=avg_response_time,
                    error_rate=error_rate,
                    active_connections=len(self.session.adapters),
                    model_loaded=True,  # We'll verify specific model in auth.py
                    details={
                        "last_response_time": response_time,
                        "connection_attempts": self.connection_attempts,
                        "total_requests": self.request_count
                    }
                )
            else:
                self.state = LLMState.ERROR
                return HealthStatus(
                    state=LLMState.ERROR,
                    last_health_check=self.last_health_check,
                    avg_response_time=0.0,
                    error_rate=1.0,
                    active_connections=0,
                    model_loaded=False,
                    details={"http_status": response.status_code}
                )
                
        except Exception as e:
            self.state = LLMState.ERROR
            self.error_count += 1
            logger.error(f"Health check failed: {e}")
            
            return HealthStatus(
                state=LLMState.ERROR,
                last_health_check=self.last_health_check,
                avg_response_time=0.0,
                error_rate=1.0,
                active_connections=0,
                model_loaded=False,
                details={"error": str(e)}
            )
    
    def ensure_connection(self) -> bool:
        """
        Ensure we have a working connection, reconnecting if necessary
        
        Returns:
            bool: True if connection is ready, False otherwise
        """
        if self.state == LLMState.READY:
            return True
            
        if self.state == LLMState.ERROR:
            logger.info("Attempting to reconnect after previous error")
            return self.connect()
        
        # If disconnected or degraded, try to reconnect
        return self.connect()
    
    def post(self, endpoint: str, data: Dict[str, Any]) -> requests.Response:
        """
        Make a POST request to Ollama API with connection management
        
        Args:
            endpoint: API endpoint (without base URL)
            data: Request data
            
        Returns:
            requests.Response: HTTP response
            
        Raises:
            LLMError: If connection or request fails
        """
        self.request_count += 1
        
        if not self.ensure_connection():
            raise LLMError(
                code=ErrorCode.CONNECTION_REFUSED,
                message="Cannot establish connection to Ollama server",
                severity=ErrorSeverity.CRITICAL,
                details={"base_url": self.config.base_url, "state": self.state.value}
            )
        
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            start_time = time.time()
            response = self.session.post(url, json=data, timeout=self.config.timeout)
            response_time = time.time() - start_time
            
            # Log slow requests
            if response_time > 10.0:
                logger.warning(f"Slow Ollama request: {response_time:.2f}s for {endpoint}")
            
            return response
            
        except requests.exceptions.Timeout:
            self.error_count += 1
            raise LLMError(
                code=ErrorCode.REQUEST_TIMEOUT,
                message="Request to Ollama timed out",
                severity=ErrorSeverity.HIGH,
                details={"timeout": self.config.timeout, "endpoint": endpoint}
            )
            
        except requests.exceptions.ConnectionError:
            self.state = LLMState.ERROR
            self.error_count += 1
            raise LLMError(
                code=ErrorCode.CONNECTION_REFUSED,
                message="Lost connection to Ollama server",
                severity=ErrorSeverity.HIGH,
                details={"base_url": self.config.base_url, "endpoint": endpoint}
            )
            
        except Exception as e:
            self.error_count += 1
            raise LLMError(
                code=ErrorCode.INVALID_REQUEST,
                message=f"Request failed: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                details={"endpoint": endpoint, "error": str(e)}
            )
    
    def close(self) -> None:
        """Cleanup and close connections"""
        if self.session:
            self.session.close()
            self.state = LLMState.DISCONNECTED
            logger.info("Ollama connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
