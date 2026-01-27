import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.document_classifier import classify_filename
from src.agents.orchestrator import DocumentOrchestrator

def test_filtering():
    test_cases = [
        ("Briefing_2024.pdf", "briefing"),
        ("Edital_Venda.docx", "edital"),
        ("Breafing_Draft.pdf", "briefing"),
        ("Editl_Publico.pdf", "edital"),
        ("Edtial_Final.pdf", "edital"),
        ("Brefing_v2.pdf", "briefing"),
        ("Briefng_Project.pdf", "briefing"),
        ("Arquivo.pdf", "invalid"),
        ("Brief.pdf", "invalid"), # "brief" is no longer a valid mandatory keyword
        ("Licitacao.pdf", "invalid"),
        ("Proposta.pdf", "invalid"),
        ("Summary_Campaign.pdf", "invalid")
    ]
    
    print("--- Testing Classifier ---")
    for filename, expected in test_cases:
        actual = classify_filename(filename)
        status = "PASSED" if actual == expected else f"FAILED (got {actual})"
        print(f"Filename: {filename:25} | Expected: {expected:10} | Actual: {actual:10} | {status}")
        assert actual == expected

    print("\n--- Testing Orchestrator (Return Message) ---")
    orchestrator = DocumentOrchestrator()
    warning_msg = "Warning: The file is not valid because its name does not comply."
    
    # Test invalid file returns warning
    result = orchestrator.analyze_document(filename="Invalid_Name.pdf", content="some content")
    print(f"Invalid Result Markdown: {result['markdown']}")
    assert result['markdown'] == warning_msg
    
    # Test unified analysis with one invalid file
    files = [
        {"name": "Briefing.pdf", "content": "valid"},
        {"name": "Invalid.pdf", "content": "invalid"}
    ]
    result_unified = orchestrator.analyze_unified(files)
    print(f"Unified Result (with invalid) Markdown: {result_unified['markdown']}")
    assert result_unified['markdown'] == warning_msg

    print("\nAll filtering tests passed successfully!")

if __name__ == "__main__":
    test_filtering()
