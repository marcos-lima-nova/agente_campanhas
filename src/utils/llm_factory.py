import os
import logging
from typing import Optional, Dict, Any
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from src.utils.ollama_adapter import OllamaGenerator

logger = logging.getLogger(__name__)

def get_llm_generator(
    model_name: Optional[str] = None,
    timeout: float = 120.0,
    generation_kwargs: Optional[Dict[str, Any]] = None
):
    """
    Factory function to return an LLM generator based on environment configuration.
    Supports OpenAI (cloud) and Ollama (local).
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_key = os.getenv("OPENAI_API_KEY")
    
    # Auto-detection/fallback logic
    if provider == "openai":
        if not openai_key or openai_key == "YOUR_VALUE_HERE":
            logger.warning("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not configured. Checking for Ollama...")
            if os.getenv("OLLAMA_BASE_URL"):
                provider = "ollama"
            else:
                logger.error("No LLM providers configured properly.")
                raise ValueError("OpenAI API key missing and no Ollama fallback found.")

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = model_name or os.getenv("OLLAMA_MODEL", "llama3.2")
        
        # Ensure url ends with /api/generate for the generator
        if not base_url.endswith("/api/generate"):
            api_url = f"{base_url.rstrip('/')}/api/generate"
        else:
            api_url = base_url
            
        logger.info(f"Initializing OllamaGenerator with model '{ollama_model}' at {api_url}")
        return OllamaGenerator(
            model=ollama_model,
            url=api_url,
            generation_kwargs=generation_kwargs,
            timeout=timeout
        )
    
    else: # Default to OpenAI
        model = model_name or "gpt-4o-mini"
        logger.info(f"Initializing OpenAIGenerator with model '{model}'")
        return OpenAIGenerator(
            api_key=Secret.from_token(openai_key),
            model=model,
            timeout=timeout,
            generation_kwargs=generation_kwargs
        )
