import requests
import logging
from typing import List, Optional, Dict, Any
from haystack import component

logger = logging.getLogger(__name__)

@component
class OllamaGenerator:
    """
    A custom Haystack 2.x component for Ollama.
    Optimized for compatibility with OpenAIGenerator's output format.
    """
    
    def __init__(
        self,
        model: str = "llama3.2",
        url: str = "http://localhost:11434/api/generate",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: float = 120.0
    ):
        self.model = model
        self.url = url
        self.generation_kwargs = generation_kwargs or {}
        self.timeout = timeout

    @component.output_types(replies=List[str])
    def run(self, prompt: str, **kwargs):
        """
        Executes the prompt against the Ollama local API.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            **self.generation_kwargs
        }
        
        # Merge runtime kwargs
        if kwargs:
            payload.update(kwargs)

        try:
            logger.info(f"Sending request to Ollama ({self.model}) at {self.url}")
            response = requests.post(self.url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            reply = data.get("response", "")
            
            if not reply:
                logger.warning("Ollama returned an empty response.")
            
            return {"replies": [reply]}
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            # Raising instead of returning None to match Haystack component behavior in pipelines
            raise RuntimeError(f"Ollama API error: {e}")

@component
class OllamaChatGenerator:
    """
    A custom Haystack 2.x component for Ollama's Chat API.
    """
    
    def __init__(
        self,
        model: str = "llama3.2",
        url: str = "http://localhost:11434/api/chat",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: float = 120.0
    ):
        self.model = model
        self.url = url
        self.generation_kwargs = generation_kwargs or {}
        self.timeout = timeout

    @component.output_types(replies=List[str])
    def run(self, messages: List[Dict[str, str]], **kwargs):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            **self.generation_kwargs
        }
        
        if kwargs:
            payload.update(kwargs)

        try:
            logger.info(f"Sending chat request to Ollama ({self.model})")
            response = requests.post(self.url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            reply = data.get("message", {}).get("content", "")
            
            return {"replies": [reply]}
            
        except Exception as e:
            logger.error(f"Ollama chat generation failed: {e}")
            raise RuntimeError(f"Ollama Chat API error: {e}")
