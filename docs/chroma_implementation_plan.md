# ChromaDB Implementation Plan

## Overview
This document outlines the plan to refactor the existing vector store implementation to use ChromaDB instead of the current Haystack InMemoryDocumentStore. The implementation will maintain backward compatibility and provide a migration path for existing data.

## Implementation Steps

### 1. ChromaDB Adapter Module
**File:** `src/vector_store/chroma_store.py`

**Purpose:** Create a wrapper around ChromaDB that exposes the same interface as the current vector store implementation.

**Key Methods:**
- `add_documents(ids, documents, metadatas, embeddings)` - Add documents to Chroma
- `query_by_embedding(embedding, top_k)` - Query using embedding vector
- `persist()` - Persist data to disk
- `load()` - Load data from disk
- `delete(ids)` - Delete documents by IDs
- `list_ids()` - List all document IDs
- `migrate_from_json(json_path)` - Migrate from legacy JSON format

### 2. Migration Script
**File:** `scripts/migrate_to_chroma.py`

**Purpose:** One-time script to migrate existing data from legacy formats to ChromaDB.

**Features:**
- Detect existing Haystack directory format
- Detect legacy JSON format (`doc_store.json`)
- Convert documents, embeddings, and metadata to Chroma format
- Handle duplicate IDs appropriately
- Log all actions for audit trail

### 3. Ingestion Pipeline Updates
**File:** `src/ingestion/run.py`

**Changes:**
- Initialize Chroma adapter
- Replace Haystack save_to_disk with Chroma persistence
- Add migration call for legacy data
- Maintain all existing chunking and embedding logic

### 4. Query Pipeline Updates
**File:** `src/rag/pipeline.py`

**Changes:**
- Initialize Chroma adapter for retrieval
- Modify query method to use Chroma for document retrieval
- Maintain existing prompt and LLM generation logic

### 5. Debug/Test Loader Updates
**File:** `src/debug_test.py`

**Changes:**
- Update document loading to use Chroma adapter
- Maintain all existing debugging functionality

### 6. Requirements Updates
**File:** `requirements.txt`

**Changes:**
- Add `chromadb[duckdb]` and related dependencies
- Remove unused vector store libraries
- Keep existing dependencies (transformers, torch, etc.)

### 7. Configuration Updates
**Files:** Environment variables and config modules

**Changes:**
- Add Chroma-specific configuration options
- Maintain existing configuration patterns

### 8. Testing
**Files:** New test files in `tests/` directory

**Tests to Implement:**
- Unit tests for Chroma adapter methods
- Integration test for full ingestion -> query flow
- Test for migration script with sample data
- Persistence test (restart application and verify data)

### 9. Documentation
**Files:** README updates

**Documentation to Include:**
- Migration process instructions
- New CLI commands for ingestion and querying
- Configuration options for Chroma
- Example usage commands

## Backward Compatibility
- CLI interfaces will remain unchanged
- API endpoints will maintain the same request/response format
- Existing configuration files will continue to work
- Legacy data will be automatically migrated on first run

## Validation Checklist
- [ ] Documents are correctly stored in Chroma collections
- [ ] Vectors can be retrieved successfully during queries
- [ ] Previously implemented pipelines continue to function
- [ ] Migration script correctly imports legacy data
- [ ] Requirements.txt contains correct dependencies
- [ ] Configuration values are properly documented
- [ ] All tests pass with the new implementation