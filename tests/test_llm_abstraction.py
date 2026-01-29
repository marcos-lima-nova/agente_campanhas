import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.llm_factory import get_llm_generator
from src.utils.ollama_adapter import OllamaGenerator
from haystack.components.generators import OpenAIGenerator

class TestLLMFactory(unittest.TestCase):

    @patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test-key"})
    def test_get_openai_generator(self):
        generator = get_llm_generator(model_name="gpt-4o")
        self.assertIsInstance(generator, OpenAIGenerator)
        self.assertEqual(generator.model, "gpt-4o")

    @patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "OLLAMA_BASE_URL": "http://localhost:11434", "OLLAMA_MODEL": "llama3"})
    def test_get_ollama_generator(self):
        generator = get_llm_generator()
        self.assertIsInstance(generator, OllamaGenerator)
        self.assertEqual(generator.model, "llama3")
        self.assertEqual(generator.url, "http://localhost:11434/api/generate")

    @patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "YOUR_VALUE_HERE", "OLLAMA_BASE_URL": "http://localhost:11434"})
    def test_fallback_to_ollama(self):
        # Should fallback to ollama if openai key is default placeholder
        generator = get_llm_generator()
        self.assertIsInstance(generator, OllamaGenerator)

    @patch("requests.post")
    def test_ollama_generator_run(self, mock_post):
        # Mock Ollama response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "This is a test reply from Ollama"}
        mock_post.return_value = mock_response

        generator = OllamaGenerator(model="llama3", url="http://localhost:11434/api/generate")
        result = generator.run(prompt="Hello")
        
        self.assertEqual(result["replies"], ["This is a test reply from Ollama"])
        mock_post.assert_called_once()

if __name__ == "__main__":
    unittest.main()
