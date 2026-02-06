"""Configuration management for the Avalon server."""

import os
from typing import Dict, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class LLMProviderConfig:
    """Configuration for a single LLM provider."""
    
    def __init__(self, name: str, api_key: str, models: List[str], base_url: Optional[str] = None):
        self.name = name
        self.api_key = api_key
        self.models = models
        self.base_url = base_url


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # MongoDB
    mongodb_uri: str = Field(default="mongodb://localhost:27017")
    mongodb_database: str = Field(default="avalon")
    
    # Global Model Configuration
    # Format: "ModelName:Provider,ModelName2:Provider2"
    available_models: str = Field(default="")

    # OpenAI
    openai_api_key: Optional[str] = Field(default=None)
    openai_base_url: Optional[str] = Field(default="https://api.openai.com/v1")
    
    # Anthropic
    anthropic_api_key: Optional[str] = Field(default=None)
    
    # DeepSeek
    deepseek_api_key: Optional[str] = Field(default=None)
    deepseek_base_url: Optional[str] = Field(default="https://api.deepseek.com")
    
    # VLLM
    vllm_api_key: Optional[str] = Field(default=None)
    vllm_base_url: Optional[str] = Field(default="http://localhost:8000/v1")
    
    class Config:
        env_file = ".env"
        extra = "ignore"
    
    def get_llm_providers(self) -> Dict[str, LLMProviderConfig]:
        """Get all configured LLM providers based on available_models."""
        providers = {}
        
        if not self.available_models:
            return providers
            
        # Parse available_models: "ModelA:ProviderA, ModelB:ProviderB"
        model_map: Dict[str, List[str]] = {}
        
        for item in self.available_models.split(","):
            if ":" in item:
                model, provider = item.split(":", 1)
                model = model.strip()
                provider = provider.strip().lower()
                
                if model and provider:
                    if provider not in model_map:
                        model_map[provider] = []
                    model_map[provider].append(model)
        
        # Create provider configs
        for provider_name, models in model_map.items():
            # Try to get credentials from explicit fields or environment variables
            env_prefix = provider_name.upper()
            
            # 1. API Key
            api_key = getattr(self, f"{provider_name}_api_key", None)
            if not api_key:
                api_key = os.getenv(f"{env_prefix}_API_KEY")
            
            # 2. Base URL
            base_url = getattr(self, f"{provider_name}_base_url", None)
            if not base_url:
                base_url = os.getenv(f"{env_prefix}_BASE_URL")
                
            if api_key:
                providers[provider_name] = LLMProviderConfig(
                    name=provider_name,
                    api_key=api_key,
                    models=models,
                    base_url=base_url
                )
                
        return providers
    
    def get_all_models(self) -> List[Dict[str, str]]:
        """Get list of all available models with their provider info."""
        models = []
        providers = self.get_llm_providers()
        
        for provider_name, provider in providers.items():
            for model in provider.models:
                models.append({
                    "provider": provider_name,
                    "model": model,
                    "display_name": f"{model}"
                })
        
        return models


settings = Settings()
