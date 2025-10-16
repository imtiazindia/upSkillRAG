"""
Authentication and model management for Ollama
Handles model verification, download, and version management
"""

import logging
import hashlib
import json
from typing import Optional, Dict, Any, List
from datetime import datetime

from .models import (
    LLMConfig, ModelType, LLMState, LLMError, 
    ErrorCode, ErrorSeverity, HealthStatus
)
from .connection import OllamaConnectionManager

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages Ollama models including verification, download, and version control
    """
    
    # Model specifications
    MODEL_SPECS = {
        ModelType.TINYLLAMA: {
            "name": "tinyllama",
            "size_gb": 0.6,
            "required_ram_gb": 2.0,
            "description": "1.1B parameter model - Fast, lightweight",
            "recommended_for": ["Testing", "Basic Q&A", "Low-resource environments"]
        },
        ModelType.PHI2: {
            "name": "phi",
            "size_gb": 1.4,
            "required_ram_gb": 3.0,
            "description": "2.7B parameter model - Good balance of speed and capability",
            "recommended_for": ["General Q&A", "Document processing", "Balanced performance"]
        },
        ModelType.LLAMA2: {
            "name": "llama2",
            "size_gb": 3.8,
            "required_ram_gb": 8.0,
            "description": "7B parameter model - High quality responses",
            "recommended_for": ["Complex reasoning", "Detailed analysis", "High-quality output"]
        }
    }
    
    def __init__(self, connection_manager: OllamaConnectionManager, config: LLMConfig):
        self.connection = connection_manager
        self.config = config
        self.available_models: List[str] = []
        self.current_model: Optional[str] = None
        self.model_details: Dict[str, Any] = {}
        
    def verify_environment(self) -> bool:
        """
        Verify that Ollama is properly installed and running
        
        Returns:
            bool: True if environment is ready, False otherwise
        """
        logger.info("Verifying Ollama environment...")
        
        # Check connection first
        if not self.connection.ensure_connection():
            logger.error("Cannot connect to Ollama server")
            return False
            
        try:
            # Get list of available models
            response = self.connection.post("/api/tags", {})
            
            if response.status_code == 200:
                data = response.json()
                self.available_models = [model["name"] for model in data.get("models", [])]
                logger.info(f"Found {len(self.available_models)} available models: {self.available_models}")
                return True
            else:
                logger.error(f"Failed to fetch models: HTTP {response.status_code}")
                return False
                
        except LLMError as e:
            logger.error(f"Error verifying environment: {e.message}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error verifying environment: {e}")
            return False
    
    def get_model_spec(self, model_type: ModelType) -> Dict[str, Any]:
        """
        Get specifications for a model type
        
        Args:
            model_type: The model type to get specs for
            
        Returns:
            Dict with model specifications
        """
        return self.MODEL_SPECS.get(model_type, {})
    
    def is_model_available(self, model_type: ModelType) -> bool:
        """
        Check if a specific model is available locally
        
        Args:
            model_type: The model type to check
            
        Returns:
            bool: True if model is available, False otherwise
        """
        model_name = self.MODEL_SPECS[model_type]["name"]
        return model_name in self.available_models
    
    def verify_model(self, model_type: ModelType) -> HealthStatus:
        """
        Verify that the specified model is available and ready
        
        Args:
            model_type: The model type to verify
            
        Returns:
            HealthStatus: Health status of the model
        """
        logger.info(f"Verifying model: {model_type.value}")
        
        if not self.connection.ensure_connection():
            return HealthStatus(
                state=LLMState.ERROR,
                last_health_check=datetime.now(),
                avg_response_time=0.0,
                error_rate=1.0,
                active_connections=0,
                model_loaded=False,
                details={"error": "No connection to Ollama"}
            )
        
        model_spec = self.get_model_spec(model_type)
        model_name = model_spec["name"]
        
        # Check if model is available
        if not self.is_model_available(model_type):
            return HealthStatus(
                state=LLMState.ERROR,
                last_health_check=datetime.now(),
                avg_response_time=0.0,
                error_rate=1.0,
                active_connections=0,
                model_loaded=False,
                details={
                    "error": f"Model {model_name} not found",
                    "available_models": self.available_models
                }
            )
        
        # Test model with a simple request
        try:
            test_prompt = "Respond with just 'OK' if you're working."
            test_data = {
                "model": model_name,
                "prompt": test_prompt,
                "stream": False
            }
            
            start_time = datetime.now()
            response = self.connection.post("/api/generate", test_data)
            
            if response.status_code == 200:
                response_data = response.json()
                response_text = response_data.get("response", "").strip()
                
                # Basic quality check
                if "OK" in response_text.upper():
                    self.current_model = model_name
                    return HealthStatus(
                        state=LLMState.READY,
                        last_health_check=datetime.now(),
                        avg_response_time=0.5,  # Placeholder
                        error_rate=0.0,
                        active_connections=1,
                        model_loaded=True,
                        details={
                            "model": model_name,
                            "test_response": response_text,
                            "response_time": (datetime.now() - start_time).total_seconds()
                        }
                    )
                else:
                    return HealthStatus(
                        state=LLMState.DEGRADED,
                        last_health_check=datetime.now(),
                        avg_response_time=0.0,
                        error_rate=0.5,
                        active_connections=1,
                        model_loaded=True,
                        details={
                            "model": model_name,
                            "warning": "Model responded but quality check failed",
                            "response": response_text
                        }
                    )
            else:
                return HealthStatus(
                    state=LLMState.ERROR,
                    last_health_check=datetime.now(),
                    avg_response_time=0.0,
                    error_rate=1.0,
                    active_connections=1,
                    model_loaded=False,
                    details={
                        "model": model_name,
                        "http_status": response.status_code,
                        "error": "Model test request failed"
                    }
                )
                
        except LLMError as e:
            return HealthStatus(
                state=LLMState.ERROR,
                last_health_check=datetime.now(),
                avg_response_time=0.0,
                error_rate=1.0,
                active_connections=0,
                model_loaded=False,
                details={
                    "model": model_name,
                    "error": e.message,
                    "error_code": e.code.value
                }
            )
    
    def download_model(self, model_type: ModelType) -> bool:
        """
        Download and install a model if not available
        
        Args:
            model_type: The model type to download
            
        Returns:
            bool: True if download successful, False otherwise
        """
        model_spec = self.get_model_spec(model_type)
        model_name = model_spec["name"]
        
        if self.is_model_available(model_type):
            logger.info(f"Model {model_name} is already available")
            return True
        
        logger.info(f"Downloading model: {model_name} ({model_spec['description']})")
        logger.info(f"Size: {model_spec['size_gb']}GB, Required RAM: {model_spec['required_ram_gb']}GB")
        
        try:
            download_data = {
                "name": model_name,
                "stream": False
            }
            
            response = self.connection.post("/api/pull", download_data)
            
            if response.status_code == 200:
                response_data = response.json()
                
                if "status" in response_data and "success" in response_data.get("status", "").lower():
                    logger.info(f"Successfully downloaded model: {model_name}")
                    # Refresh available models list
                    self.verify_environment()
                    return True
                else:
                    logger.error(f"Download failed: {response_data}")
                    return False
            else:
                logger.error(f"Download request failed: HTTP {response.status_code}")
                return False
                
        except LLMError as e:
            logger.error(f"Error downloading model {model_name}: {e.message}")
            return False
    
    def get_model_info(self, model_type: ModelType) -> Dict[str, Any]:
        """
        Get detailed information about a model
        
        Args:
            model_type: The model type to get info for
            
        Returns:
            Dict with model information
        """
        model_spec = self.get_model_spec(model_type)
        model_name = model_spec["name"]
        
        info = {
            "specification": model_spec,
            "available": self.is_model_available(model_type),
            "current": self.current_model == model_name
        }
        
        if self.is_model_available(model_type):
            # Get model details from Ollama
            try:
                response = self.connection.post("/api/show", {"name": model_name})
                if response.status_code == 200:
                    info["details"] = response.json()
            except LLMError:
                pass  # Ignore errors in detailed info
        
        return info
    
    def get_available_models_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all available models
        
        Returns:
            List of model information dictionaries
        """
        models_info = []
        
        for model_type in ModelType:
            model_info = self.get_model_info(model_type)
            models_info.append({
                "type": model_type.value,
                **model_info
            })
        
        return models_info
    
    def get_update_instructions(self, model_type: ModelType) -> Dict[str, Any]:
        """
        Get instructions for updating a model
        
        Args:
            model_type: The model type to update
            
        Returns:
            Dict with update instructions and information
        """
        model_spec = self.get_model_spec(model_type)
        model_name = model_spec["name"]
        
        instructions = {
            "model": model_name,
            "update_command": f"ollama pull {model_name}",
            "description": f"Update {model_spec['description']} to latest version",
            "estimated_size": f"{model_spec['size_gb']}GB",
            "steps": [
                "1. Stop any running applications using this model",
                "2. Open terminal/command prompt",
                f"3. Run: ollama pull {model_name}",
                "4. Wait for download to complete",
                "5. Restart your application"
            ],
            "notes": [
                "Updating will replace the current model version",
                "Existing model files will be overwritten",
                "Internet connection required for download",
                "May take several minutes depending on your connection"
            ]
        }
        
        return instructions
