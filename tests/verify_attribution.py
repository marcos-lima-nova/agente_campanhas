import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.rag.pipeline import RAGPipeline
from haystack import Document

def test_source_filtering():
    # Mock DocumentStore and components
    mock_store = MagicMock()
    
    with patch('src.rag.pipeline.Pipeline') as mock_pipeline_class, \
         patch('src.rag.pipeline.OpenAIGenerator') as mock_gen_class, \
         patch('src.rag.pipeline.SentenceTransformersTextEmbedder') as mock_embed_class:
        
        mock_pipeline = mock_pipeline_class.return_value
        
        # Setup mock retrieval results
        doc1 = Document(content="Content from doc 1", meta={"filename": "doc1.pdf"})
        doc2 = Document(content="Content from doc 2", meta={"filename": "doc2.pdf"})
        
        # Mock pipeline.run output
        # Case 1: LLM only cites SOURCE_0
        mock_pipeline.run.return_value = {
            "llm": {"replies": ["This info comes from [SOURCE_0]."]},
            "retriever": {"documents": [doc1, doc2]}
        }
        
        pipeline = RAGPipeline(document_store=mock_store)
        result = pipeline.query("test question")
        
        print(f"Answer: {result['answer']}")
        print(f"Sources: {[s['filename'] for s in result['sources']]}")
        
        assert len(result['sources']) == 1
        assert result['sources'][0]['filename'] == "doc1.pdf"
        print("Test 1 Passed: Only cited source included.")

        # Case 2: LLM cites both
        mock_pipeline.run.return_value = {
            "llm": {"replies": ["Info from [SOURCE_0] and [SOURCE_1]."]},
            "retriever": {"documents": [doc1, doc2]}
        }
        result = pipeline.query("test question")
        assert len(result['sources']) == 2
        print("Test 2 Passed: Both cited sources included.")

        # Case 3: LLM cites none
        mock_pipeline.run.return_value = {
            "llm": {"replies": ["No context used."]},
            "retriever": {"documents": [doc1, doc2]}
        }
        result = pipeline.query("test question")
        assert len(result['sources']) == 0
        print("Test 3 Passed: No cited sources included.")

if __name__ == "__main__":
    try:
        test_source_filtering()
        print("\nAll verification tests completed successfully!")
    except Exception as e:
        print(f"\nVerification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
