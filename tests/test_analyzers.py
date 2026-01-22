import json
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from unittest.mock import MagicMock, AsyncMock
from fastapi import Request
from src.ui.server import chat_completions, get_rag, get_orchestrator

async def test_analyzer_fallback():
    print("Testing analyzer fallback (no files)...")
    # Mock Request
    request = MagicMock(spec=Request)
    request.json = AsyncMock(return_value={
        "model": "gpt-4o-mini-analyzer",
        "messages": [{"role": "user", "content": "What is in the file?"}],
        "files": []
    })
    
    # Mock dependencies
    rag = MagicMock()
    orchestrator = MagicMock()
    
    response = await chat_completions(request, rag, orchestrator)
    print(f"Response: {response['choices'][0]['message']['content']}")
    assert "requer um arquivo anexado" in response['choices'][0]['message']['content']
    print("Pass: Fallback working message returned.")

async def test_unified_analyzer_flow():
    print("\nTesting unified analyzer flow...")
    # Mock Request with 2 files
    request = MagicMock(spec=Request)
    request.json = AsyncMock(return_value={
        "model": "unified-analyzer-agent",
        "messages": [{"role": "user", "content": "Analyze these"}],
        "files": [
            {"id": "file1", "name": "briefing.pdf"},
            {"id": "file2", "name": "edital.pdf"}
        ]
    })
    
    # Mock dependencies
    rag = MagicMock()
    orchestrator = MagicMock()
    orchestrator.analyze_unified.return_value = {
        "markdown": "# Unified Analysis Results",
        "filename": "unified.md",
        "doc_type": "unified"
    }
    
    # Mock fetch_file_from_webui in server module
    import src.ui.server as server
    server.fetch_file_from_webui = MagicMock(return_value=b"fake bytes")
    
    response = await chat_completions(request, rag, orchestrator)
    print(f"Response Content: {response['choices'][0]['message']['content']}")
    assert response['model'] == "unified-analyzer-agent"
    assert "Unified Analysis Results" in response['choices'][0]['message']['content']
    print("Pass: Unified analyzer flow routed correctly.")

if __name__ == "__main__":
    asyncio.run(test_analyzer_fallback())
    asyncio.run(test_unified_analyzer_flow())
