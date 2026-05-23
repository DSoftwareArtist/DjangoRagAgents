# Architecture

## Overview

This project is a monolithic Django application with a small service layer for Retrieval-Augmented Generation. It combines:

- Django views and templates for the user interface
- Django models for document, chunk, and query persistence
- PostgreSQL with `pgvector` for vector storage and similarity search
- Local embedding generation with `sentence-transformers`
- Groq-hosted LLM inference through the OpenAI Python client

At a high level, the app turns uploaded files into embeddings, retrieves the closest chunks for a question, and asks a chat model to answer using those chunks as context.

## High-Level Component Map

### Django project layer

- [manage.py](/Users/reamonsumapig/Random/DjangoRag/manage.py) boots Django commands.
- [django_rag/settings.py](/Users/reamonsumapig/Random/DjangoRag/django_rag/settings.py) configures apps, database, media, and model settings.
- [django_rag/urls.py](/Users/reamonsumapig/Random/DjangoRag/django_rag/urls.py) mounts the app routes.

### Application layer

- [rag_app/views.py](/Users/reamonsumapig/Random/DjangoRag/rag_app/views.py) handles HTML and JSON endpoints.
- [rag_app/forms.py](/Users/reamonsumapig/Random/DjangoRag/rag_app/forms.py) defines Django forms, although the current templates render fields manually.
- [templates/](/Users/reamonsumapig/Random/DjangoRag/templates) contains the Bootstrap-based UI.

### Domain and persistence layer

- [rag_app/models.py](/Users/reamonsumapig/Random/DjangoRag/rag_app/models.py) defines `Document`, `DocumentChunk`, and `QueryHistory`.
- [rag_app/migrations/0000_enable_pgvector.py](/Users/reamonsumapig/Random/DjangoRag/rag_app/migrations/0000_enable_pgvector.py) enables the PostgreSQL `vector` extension.
- [rag_app/migrations/0001_initial.py](/Users/reamonsumapig/Random/DjangoRag/rag_app/migrations/0001_initial.py) creates the schema.

### RAG service layer

[rag_app/services/rag_service.py](/Users/reamonsumapig/Random/DjangoRag/rag_app/services/rag_service.py) contains three cooperating classes:

- `EmbeddingService`: lazy singleton wrapper around `SentenceTransformer`
- `TextProcessor`: file reading, PDF extraction, and chunking
- `RAGService`: orchestration for ingestion, retrieval, prompting, and answer generation

The query path now has an initial LangGraph workflow for stateful orchestration, while ingestion still runs as direct service logic.

### Operations layer

- [rag_app/management/commands/ingest_documents.py](/Users/reamonsumapig/Random/DjangoRag/rag_app/management/commands/ingest_documents.py) ingests files from disk without using the web UI.

## Data Flow

### 1. Document upload and ingestion

Entry point:

- `DocumentUploadView.post()`
- `api_upload_document()`
- `ingest_documents` management command

Flow:

1. A `Document` row is created with a file path under `documents/`.
2. `RAGService.ingest_document(document)` resolves `document.file.path`.
3. `TextProcessor.process_file()` selects handling based on file extension.
4. `.txt` files are read directly.
5. `.pdf` files are parsed with `PyPDF2.PdfReader`.
6. Full extracted text is stored in `Document.content`.
7. `RecursiveCharacterTextSplitter` chunks the text using:
   - `CHUNK_SIZE = 1000`
   - `CHUNK_OVERLAP = 200`
8. Each chunk is embedded with `sentence-transformers/all-MiniLM-L6-v2`.
9. A `DocumentChunk` row is created for every chunk.
10. The document is marked `processed=True`.

### 2. Retrieval and answering

Entry point:

- `RAGQueryView.post()`
- `api_query()`

Flow:

1. The query text is embedded with the same embedding model.
2. A LangGraph graph starts with retrieval and then branches based on whether any chunks were found.
3. `DocumentChunk` rows are annotated with `L2Distance('embedding', query_embedding)`.
4. The nearest 5 chunks are selected.
5. Similarity shown to users is computed as `1 / (1 + distance)`.
6. Chunk contents are concatenated into a prompt:
   - `Context: ...`
   - `Question: ...`
   - `Answer:`
7. The prompt is sent to Groq using the OpenAI client with:
   - `base_url = "https://api.groq.com/openai/v1"`
   - `model = "llama-3.1-8b-instant"`
8. A final graph node formats the source payload for the UI and API.
9. The answer and summarized sources are saved to `QueryHistory`.
10. The response is rendered in HTML or returned as JSON.

## Persistence Design

### `Document`

Purpose:

- Tracks uploaded files and whether they have been processed.

Notable behavior:

- The full extracted text is duplicated into the database in `content`.
- The original file remains on disk under `MEDIA_ROOT`.

### `DocumentChunk`

Purpose:

- Stores retrieval-sized text segments and vector embeddings.

Notable behavior:

- Embedding dimension is fixed at `384`, which must stay aligned with the chosen sentence-transformer model.
- Rows are ordered by `(document, chunk_index)` for deterministic reconstruction.

### `QueryHistory`

Purpose:

- Stores prior questions, generated answers, and source metadata.

Notable behavior:

- `sources` is a JSON payload, not a normalized relation.

## Interface Surface

### HTML endpoints

- `GET /` -> document list
- `GET /documents/upload/` -> upload form
- `POST /documents/upload/` -> save and ingest a document
- `GET /rag/query/` -> query page with recent history
- `POST /rag/query/` -> retrieve and answer
- `GET /rag/answer/<pk>/` -> saved answer view

### JSON endpoints

- `POST /api/documents/upload/`
- `POST /api/rag/query/`

The JSON API is intentionally thin and reuses the same service layer as the HTML views.

## Runtime Dependencies

### Database

- PostgreSQL is required.
- The app assumes the `vector` extension can be enabled.

### Local model inference

- Embeddings are generated inside the Django process.
- The first embedding request will load the transformer model, which can add startup latency and memory cost.

### External inference

- Answer generation depends on a reachable Groq API and a valid `GROQ_API_KEY`.

## Current Design Characteristics

### Strengths

- Small codebase with a clear flow from upload to answer
- Separation between views and RAG orchestration
- Query orchestration now has explicit graph nodes and routing
- Supports both UI-driven and command-line ingestion
- Uses PostgreSQL directly for retrieval without adding another vector store service

### Constraints

- Upload processing is synchronous and request-bound
- LangGraph currently covers query orchestration, not ingestion
- No background jobs, task queue, or retries
- No deduplication or cleanup of old `DocumentChunk` rows during reprocessing
- Prompt construction is minimal and has no guardrails, citations, or answer formatting policy
- No pagination or filtering on document history
- No authentication around upload or query endpoints

## Extension Points

If you want to evolve the system, the most natural places are:

- `RAGService.generate_answer()` for better prompts, citations, or structured outputs
- `RAGService.retrieve_similar_chunks()` for alternative ranking strategies or metadata filtering
- `TextProcessor` for support beyond `.txt` and `.pdf`
- the upload flow for async/background ingestion
- the data model for tags, ownership, or document collections

## Practical Caveats

- The management command uses `Document.objects.get_or_create(title=title, ...)`, so title collisions can affect reuse behavior.
- The upload path validates file extensions in the view layer, while the service layer trusts the file path it receives.
- `api_query()` parses raw JSON from `request.body` and assumes valid JSON input.
- Existing tests cover models and basic endpoints, but they do not isolate external model/API dependencies.

## Summary

The current architecture is a straightforward Django RAG monolith: PostgreSQL stores both source metadata and vectors, the service layer handles ingestion and retrieval, and Groq provides the final answer generation step. It is a solid base for experimentation and small deployments, with the main scaling pressure coming from synchronous ingestion and in-process embedding generation.
