"""Migration utilities for re-indexing existing documents with Late Chunking.

Provides a function to iterate over documents already stored in a
:class:`~src.vector_store.chroma_store.ChromaAdapter`, re-embed them using
:class:`~src.indexing.late_chunker.LateChunker`, and upsert the new
embeddings back into the store.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from src.indexing.late_chunker import LateChunker
from src.vector_store.chroma_store import ChromaAdapter

logger = logging.getLogger(__name__)


def migrate_existing_embeddings(
    store: ChromaAdapter,
    late_chunker: LateChunker,
    batch_size: int = 50,
    dry_run: bool = False,
) -> int:
    """Re-index existing documents in *store* using Late Chunking.

    The function:

    1. Reads all document IDs currently in the collection.
    2. Fetches their text content in batches.
    3. Groups contiguous chunks that share the same ``filename`` metadata
       back into a single document text.
    4. Runs :meth:`LateChunker.process_document` to generate new embeddings.
    5. Deletes the old entries and inserts the new ones.

    Args:
        store: A :class:`ChromaAdapter` pointing to the target collection.
        late_chunker: An initialised :class:`LateChunker` instance.
        batch_size: Number of IDs to fetch per batch from Chroma.
        dry_run: If ``True``, log what *would* happen without writing.

    Returns:
        The total number of new chunks written to the store.
    """
    all_ids = store.list_ids()
    if not all_ids:
        logger.info("No documents found in store; nothing to migrate.")
        return 0

    logger.info("Starting migration of %d existing entries", len(all_ids))

    # --- Fetch all existing documents ---
    # ChromaDB get() supports include parameter
    data = store.collection.get(
        ids=all_ids,
        include=["documents", "metadatas"],
    )

    documents: List[str] = data.get("documents", [])
    metadatas: List[dict] = data.get("metadatas", [])
    ids: List[str] = data.get("ids", [])

    if not documents:
        logger.warning("Fetched IDs but no document texts found; aborting migration.")
        return 0

    # --- Group chunks by source file ---
    # We reconstruct the original document text by joining chunks with the
    # same ``filename`` metadata key.
    file_groups: dict[str, dict] = {}  # filename -> {text_parts, metadata}

    for doc_id, text, meta in zip(ids, documents, metadatas):
        fname = (meta or {}).get("filename", doc_id)
        if fname not in file_groups:
            file_groups[fname] = {
                "text_parts": [],
                "metadata": {k: v for k, v in (meta or {}).items()},
                "old_ids": [],
            }
        file_groups[fname]["text_parts"].append(text or "")
        file_groups[fname]["old_ids"].append(doc_id)

    logger.info("Grouped into %d unique source files", len(file_groups))

    total_written = 0

    for fname, group in file_groups.items():
        full_text = " ".join(group["text_parts"])
        base_meta = group["metadata"]
        old_ids = group["old_ids"]

        if not full_text.strip():
            logger.warning("Skipping empty document group for %s", fname)
            continue

        logger.info(
            "Re-indexing %s (%d old chunks, %d chars)",
            fname,
            len(old_ids),
            len(full_text),
        )

        if dry_run:
            logger.info("[DRY RUN] Would re-index %s", fname)
            continue

        # Generate new chunks with late chunking
        new_docs = late_chunker.process_document(full_text, metadata=base_meta)

        if not new_docs:
            logger.warning("Late chunker produced no chunks for %s; keeping originals.", fname)
            continue

        # Build new IDs
        file_hash = base_meta.get("hash", fname)
        new_ids = [f"{file_hash}_lc_{i}" for i in range(len(new_docs))]
        new_texts = [d.content for d in new_docs]
        new_metas = [d.meta for d in new_docs]
        new_embeddings = [d.embedding for d in new_docs]

        # Delete old entries
        try:
            store.delete(ids=old_ids)
        except Exception as e:
            logger.error("Failed to delete old entries for %s: %s", fname, e)
            continue

        # Insert new entries
        try:
            store.add_documents(
                ids=new_ids,
                documents=new_texts,
                metadatas=new_metas,
                embeddings=new_embeddings,
            )
            total_written += len(new_docs)
            logger.info(
                "Migrated %s: %d old chunks → %d new chunks",
                fname,
                len(old_ids),
                len(new_docs),
            )
        except Exception as e:
            logger.error("Failed to write new entries for %s: %s", fname, e)

    # Persist changes
    if not dry_run:
        try:
            store.persist()
        except Exception as e:
            logger.warning("Persist call failed (may be auto-persisted): %s", e)

    logger.info("Migration complete. Total new chunks written: %d", total_written)
    return total_written
