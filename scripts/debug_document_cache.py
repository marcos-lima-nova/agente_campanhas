#!/usr/bin/env python
"""
Debug script to trace document caching and re-analysis behavior.

Run this script to simulate the flow and identify why analysis might be
re-running on follow-up questions.

Usage:
    python scripts/debug_document_cache.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import time
from unittest.mock import MagicMock
from src.agents.conversation_orchestrator import ConversationOrchestrator, _compute_document_id
from src.agents.intent_detector import Intent
from src.utils.session_manager import InMemoryBackend, SessionManager


def main():
    print("=" * 60)
    print("DOCUMENT CACHE DEBUG SCRIPT")
    print("=" * 60)
    
    # Setup
    session_manager = SessionManager(backend=InMemoryBackend(), ttl=3600)
    
    doc_mock = MagicMock()
    doc_mock.analyze_document.return_value = {
        "markdown": "# Análise\n\nCliente: ACME\n\n## Palavras-chave\n\n- marketing\n- digital",
        "filename": "briefing_analysis.md",
        "doc_type": "briefing",
        "timestamp": time.time(),
        "analyzer_id": "briefing-analyzer",
    }
    
    rag_mock = MagicMock()
    rag_mock.query_with_context.return_value = {
        "answer": "Based on the analysis, here are similar campaigns...",
        "sources": [],
    }
    
    orchestrator = ConversationOrchestrator(
        document_orchestrator=doc_mock,
        rag_pipeline=rag_mock,
        session_manager=session_manager,
    )
    
    # Sample file
    sample_file = {
        "name": "Briefing_ACME.pdf",
        "id": "file-001",
        "bytes": b"fake pdf bytes for ACME briefing",
    }
    
    # Compute expected document_id
    expected_doc_id = _compute_document_id(sample_file["bytes"], sample_file["name"])
    print(f"\nExpected document_id: {expected_doc_id}")
    
    # Turn 1: Initial analysis
    print("\n" + "-" * 60)
    print("TURN 1: Initial document upload + analysis")
    print("-" * 60)
    
    r1 = orchestrator.handle_message(
        message="",
        files=[sample_file],
    )
    
    print(f"Response intent: {r1.intent.value}")
    print(f"Response session_id: {r1.session_id}")
    print(f"Analysis ran: {doc_mock.analyze_document.called}")
    
    session = session_manager.get_session(r1.session_id)
    print(f"\nSession state after Turn 1:")
    print(f"  - documents keys: {list(session.documents.keys())}")
    print(f"  - active_document_id: {session.active_document_id}")
    print(f"  - last_analysis present: {session.last_analysis is not None}")
    print(f"  - conversation_history length: {len(session.conversation_history)}")
    
    if expected_doc_id in session.documents:
        print(f"  [OK] Document {expected_doc_id} IS in cache")
    else:
        print(f"  [FAIL] Document {expected_doc_id} NOT in cache!")
    
    # Reset mock for next turn
    doc_mock.reset_mock()
    
    # Turn 2: Follow-up question WITH file re-sent (Open WebUI behavior)
    print("\n" + "-" * 60)
    print("TURN 2: Follow-up question with SAME file re-sent")
    print("-" * 60)
    
    r2 = orchestrator.handle_message(
        message="quais campanhas similares?",
        session_id=r1.session_id,
        files=[sample_file],  # Same file re-sent
    )
    
    print(f"Response intent: {r2.intent.value}")
    print(f"Analysis ran: {doc_mock.analyze_document.called}")
    print(f"QA ran: {rag_mock.query_with_context.called}")
    
    session = session_manager.get_session(r2.session_id)
    print(f"\nSession state after Turn 2:")
    print(f"  - documents keys: {list(session.documents.keys())}")
    print(f"  - active_document_id: {session.active_document_id}")
    
    # Verify expected behavior
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    if doc_mock.analyze_document.called:
        print("[FAIL] Analysis ran on follow-up (should have been cached)")
    else:
        print("[PASS] Analysis did NOT run on follow-up (cached)")
    
    if r2.intent == Intent.QA:
        print("[PASS] Intent was QA (correct)")
    else:
        print(f"[FAIL] Intent was {r2.intent.value} (expected QA)")
    
    if expected_doc_id in session.documents:
        print("[PASS] Document is in cache")
    else:
        print("[FAIL] Document NOT in cache")
    
    # Turn 3: Follow-up question WITHOUT file
    print("\n" + "-" * 60)
    print("TURN 3: Follow-up question WITHOUT file")
    print("-" * 60)
    
    doc_mock.reset_mock()
    rag_mock.reset_mock()
    
    r3 = orchestrator.handle_message(
        message="O que diz o briefing sobre o público-alvo?",
        session_id=r1.session_id,
        files=[],  # No file this time
    )
    
    print(f"Response intent: {r3.intent.value}")
    print(f"Analysis ran: {doc_mock.analyze_document.called}")
    print(f"QA ran: {rag_mock.query_with_context.called}")
    
    if doc_mock.analyze_document.called:
        print("[FAIL] Analysis ran on follow-up without file")
    else:
        print("[PASS] Analysis did NOT run on follow-up without file")
    
    if r3.intent == Intent.QA:
        print("[PASS] Intent was QA (correct)")
    else:
        print(f"[FAIL] Intent was {r3.intent.value} (expected QA)")
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
