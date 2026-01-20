import os
import re
from pathlib import Path
from typing import List, Optional
import pypdf
import docx
from haystack import Document
import io

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, file_source: Path | str | bytes, filename: Optional[str] = None) -> str:
        """Extracts text from PDF or DOCX. Source can be a path or bytes."""
        import io
        
        if isinstance(file_source, (Path, str)):
            file_path = Path(file_source)
            suffix = file_path.suffix.lower()
            with open(file_path, "rb") as f:
                stream = io.BytesIO(f.read())
        elif isinstance(file_source, bytes):
            stream = io.BytesIO(file_source)
            if not filename:
                # We need a filename or at least an extension to know how to process
                raise ValueError("filename must be provided when passing bytes to extract_text")
            suffix = Path(filename).suffix.lower()
        else:
            raise ValueError("Unsupported file source type")

        if suffix == ".pdf":
            return self._extract_pdf(stream)
        elif suffix == ".docx":
            return self._extract_docx(stream)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _extract_pdf(self, stream: io.BytesIO) -> str:
        text = ""
        reader = pypdf.PdfReader(stream)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def _extract_docx(self, stream: io.BytesIO) -> str:
        doc = docx.Document(stream)
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
