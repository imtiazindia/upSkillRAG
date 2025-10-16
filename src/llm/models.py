"""
Data models and schemas for LLM layer
Defines contracts for requests, responses, errors, and telemetry
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List

class LLMProvider(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"

class ModelType(Enum):
    TINYLLAMA = "tinyllama"
    PHI2 = "phi2"
    LLAMA2 = "llama2"

class LLMState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    READY = "ready"
    ERROR = "error"
    DEGRADED = "degraded"  # Connected but slow/high latency

class ErrorSeverity(Enum):
    LOW = "low"       # Minor issue, operation continues
    MEDIUM = "medium" # Some features affected
    HIGH = "high"     # Major functionality broken
    CRITICAL = "critical" # System unusable

class ErrorCode(Enum):
    # Connection Errors
    OLLAMA_NOT_RUNNING = "ollama_not_running"
    CONNECTION_TIMEOUT = "connection_timeout"
    CONNECTION_REFUSED = "connection_refused"
    
    # Model Errors
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_LOAD_FAILED = "model_load_failed"
    MODEL_OUT_OF_MEMORY = "model_out_of_memory"
    
    # Request Errors
    REQUEST_TIMEOUT = "request_timeout"
    INVALID_REQUEST = "invalid_request"
    RATE_LIMITED = "rate_limited"
    
    # Response Errors
    EMPTY_RESPONSE = "empty_response"
    MALFORMED_RESPONSE = "malformed_response"
    QUALITY_ISSUE = "quality_issue"

@dataclass
class LLMRequest:
    """Schema for LLM requests"""
    prompt: str
    model: ModelType = ModelType.TINYLLAMA
    temperature: float = 0.1
    max_tokens: int = 512
    stream: bool = False
    context: Optional[str] = None  # For RAG context
    request_id: Optional[str] = None

@dataclass
class LLMResponse:
    """Schema for LLM responses"""
    content: str
    request_id: str
    model_used: ModelType
    tokens_used: int
    processing_time: float  # seconds
    timestamp: datetime
    context_used: Optional[str] = None

@dataclass
class LLMError:
    """Schema for LLM errors"""
    code: ErrorCode
    message: str
    severity: ErrorSeverity
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    request_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class TelemetryData:
    """Schema for performance and usage telemetry"""
    request_id: str
    model: ModelType
    start_time: datetime
    end_time: datetime
    processing_time: float
    tokens_input: int
    tokens_output: int
    cpu_time: float
    memory_used: int  # bytes
    state: LLMState
    error: Optional[LLMError] = None
    cache_hit: bool = False

@dataclass
class HealthStatus:
    """Schema for LLM health checks"""
    state: LLMState
    last_health_check: datetime
    avg_response_time: float
    error_rate: float  # 0.0 to 1.0
    active_connections: int
    model_loaded: bool
    details: Optional[Dict[str, Any]] = None

@dataclass
class CacheEntry:
    """Schema for response cache entries"""
    prompt_hash: str
    response: LLMResponse
    created_at: datetime
    access_count: int
    last_accessed: datetime

@dataclass
class LLMConfig:
    """Configuration model for LLM settings"""
    provider: LLMProvider
    model: ModelType
    base_url: str = "http://localhost:11434"
    timeout: int = 30
    max_retries: int = 3
    health_check_frequency: int = 10  # Every N requests
    cache_ttl: int = 3600  # 1 hour in seconds
    cache_max_size: int = 1000
    enable_telemetry: bool = True
    
    # Performance tuning
    temperature: float = 0.1
    max_tokens: int = 512
    top_p: float = 0.9
