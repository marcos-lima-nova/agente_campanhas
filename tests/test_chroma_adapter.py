import os
import tempfile
from pathlib import Path

from src.vector_store.chroma_store import ChromaAdapter


def test_add_and_query():
    tmpdir = Path(tempfile.mkdtemp())
    adapter = ChromaAdapter(persist_directory=tmpdir)

    ids = ["id1", "id2"]
    docs = ["Hello world", "Goodbye world"]
    metas = [{"filename": "a.txt"}, {"filename": "b.txt"}]
    # Use simple 3-dim vectors for testing
    embeddings = [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]

    adapter.add_documents(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    adapter.persist()

    # Query using one of the embeddings
    res = adapter.query_by_embedding(embeddings[0], top_k=1)
    assert res["ids"]
    assert res["documents"]
    assert res["metadatas"]

    # Clean up
    adapter.delete()

