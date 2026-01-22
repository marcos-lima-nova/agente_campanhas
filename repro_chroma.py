import chromadb
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
path = Path(os.getenv("VECTOR_STORE_PATH", "vectorstore/")).resolve()
print(f"Connecting to: {path}")

try:
    client = chromadb.PersistentClient(path=str(path))
    print("Success: Connected to ChromaDB")
    col = client.get_or_create_collection("documents")
    print(f"Collection count: {col.count()}")
except Exception as e:
    print(f"Caught exception: {e}")
except BaseException as be:
    print(f"Caught base exception (including panics): {be}")
