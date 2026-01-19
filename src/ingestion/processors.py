import os
import re
from pathlib import Path
from typing import List, Optional
import pypdf
import docx
from haystack import Document

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, file_path: Path) -> str:
        """Extracts text from PDF or DOCX."""
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(file_path)
        elif suffix == ".docx":
            return self._extract_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _extract_pdf(self, file_path: Path) -> str:
        text = ""
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _extract_docx(self, file_path: Path) -> str:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def create_chunks(self, text: str, metadata: dict) -> List[Document]:
        """Simple chunking logic with overlap."""
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        chunks = []
        
        # Use a simple word-based chunking for efficiency
        # In a real scenario, you might want to use Haystack's DocumentSplitter
        # but here we implement a manual one as requested for flexibility
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            if chunk_text:
                chunks.append(Document(content=chunk_text, meta=metadata.copy()))
        
        return chunks

    @staticmethod
    def infer_campaign(filename: str) -> str:
        """Infers campaign name from filename (e.g., 'Summer2024_promo.pdf' -> 'Summer2024')."""
        match = re.search(r'^([^_.-]+)', filename)
        return match.group(1) if match else "General"
