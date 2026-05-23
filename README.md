# Django RAG System

A Django-based Retrieval-Augmented Generation (RAG) app for uploading `.txt` and `.pdf` files, storing embeddings in PostgreSQL with `pgvector`, and answering questions with Groq's OpenAI-compatible chat API.

<img width="986" height="746" alt="Screenshot 2026-03-19 at 6 01 46 PM" src="https://github.com/user-attachments/assets/20f7845b-d875-4c1a-81c3-8be9f4073d7a" />

## What It Does

- Upload text and PDF documents through the web UI or JSON API.
- Extract raw text from each file.
- Split text into chunks with overlap for retrieval.
- Generate 384-dimensional embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Store chunk embeddings in PostgreSQL using `pgvector`.
- Retrieve the closest chunks with vector similarity search.
- Send retrieved context to Groq's `llama-3.1-8b-instant` model to generate an answer.
- Save query history and returned sources for later viewing.

## Stack

- Django 4.2
- PostgreSQL
- `pgvector`
- `langgraph`
- `sentence-transformers`
- `langchain-text-splitters`
- Groq via the OpenAI Python client
- `PyPDF2`
- `python-dotenv`

Note: the app now uses LangGraph for the initial query orchestration path, while keeping the rest of the service layer and Django views intact.

## Project Layout

```text
DjangoRag/
├── django_rag/                         # Django project config
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
├── rag_app/                            # Main app
│   ├── management/commands/
│   │   └── ingest_documents.py         # Batch ingestion command
│   ├── migrations/
│   │   ├── 0000_enable_pgvector.py     # Enables PostgreSQL vector extension
│   │   └── 0001_initial.py             # Creates core tables
│   ├── services/
│   │   └── rag_service.py              # Text processing, embeddings, retrieval, LLM calls
│   ├── forms.py
│   ├── models.py
│   ├── urls.py
│   └── views.py
├── templates/
│   ├── base.html
│   └── rag_app/                        # Document, query, and result templates
├── documents/                          # Uploaded files and media root
├── requirements.txt
├── README.md
└── Architecture.md
```

## Data Model

### `Document`

Stores uploaded file metadata, extracted text, and processing state.

- `title`
- `file`
- `content`
- `uploaded_at`
- `processed`

### `DocumentChunk`

Stores chunked document content plus a 384-dimensional embedding.

- `document`
- `content`
- `chunk_index`
- `embedding`
- `created_at`

### `QueryHistory`

Stores user questions, generated answers, and source snippets returned to the UI/API.

- `query`
- `answer`
- `sources`
- `created_at`

## Request Flow

### Document ingestion

1. A user uploads a `.txt` or `.pdf` file.
2. `DocumentUploadView` saves the file as a `Document`.
3. `RAGService.ingest_document()` reads the file from disk.
4. `TextProcessor` extracts raw text.
5. Text is chunked using `RecursiveCharacterTextSplitter`.
6. `EmbeddingService` generates an embedding for each chunk.
7. Each chunk is saved as a `DocumentChunk`.
8. The document is marked as processed.

### Query answering

1. A user submits a question from the UI or API.
2. `RAGService.retrieve_similar_chunks()` embeds the query.
3. A LangGraph workflow routes either to a no-results response or to answer generation.
4. `pgvector` `L2Distance` ranks the closest chunks.
5. Top chunks are concatenated into a prompt.
6. Groq generates the answer with `llama-3.1-8b-instant`.
7. The answer and sources are saved in `QueryHistory`.

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Create PostgreSQL database and enable `pgvector`

Create a PostgreSQL database, then make sure the `vector` extension can be enabled by migrations.

Example:

```sql
CREATE DATABASE rag_db;
```

The app migration `rag_app/migrations/0000_enable_pgvector.py` runs:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3. Configure environment variables

Copy `.env.template` to `.env` and set values:

```env
DB_NAME=rag_db
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432

GROQ_API_KEY=your_groq_api_key_here

SECRET_KEY=your_django_secret_key_here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
```

### 4. Run migrations

```bash
python manage.py migrate
```

### 5. Start the server

```bash
python manage.py runserver
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Docker

You can run the app locally with Docker Compose and a dedicated PostgreSQL + `pgvector` container.

### 1. Create your `.env`

Copy [.env.template](/Users/reamonsumapig/Random/DjangoRag/.env.template) to `.env` and set at least:

```env
DB_NAME=rag_db
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_django_secret_key_here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
```

Note: in Docker, `docker-compose.yml` overrides `DB_HOST` to `db`, so you can keep `localhost` in your local `.env` for non-Docker runs.

### 2. Build and start the containers

```bash
docker compose up --build
```

This starts:

- `web` on [http://127.0.0.1:8000](http://127.0.0.1:8000)
- `db` on `localhost:5432`

The web container runs migrations automatically on startup.

### 3. Stop the stack

```bash
docker compose down
```

To also remove the Postgres data volume:

```bash
docker compose down -v
```

### 4. Useful Docker commands

Run Django tests inside the app container:

```bash
docker compose exec web python manage.py test
```

Open a Django shell:

```bash
docker compose exec web python manage.py shell
```

Inspect the database with `psql`:

```bash
docker compose exec db psql -U postgres -d rag_db
```

### Docker files

- [Dockerfile](/Users/reamonsumapig/Random/DjangoRag/Dockerfile) builds the Django app image.
- [docker-compose.yml](/Users/reamonsumapig/Random/DjangoRag/docker-compose.yml) defines the app and database services.
- [.dockerignore](/Users/reamonsumapig/Random/DjangoRag/.dockerignore) keeps the Docker build context small.

## Usage

### Web UI

- `/` shows uploaded documents and processing status.
- `/documents/upload/` uploads and processes a document.
- `/rag/query/` asks a question and shows recent query history.
- `/rag/answer/<id>/` shows a saved result.

### API

#### Upload a document

`POST /api/documents/upload/`

Multipart form fields:

- `title`
- `file`

Returns:

```json
{
  "success": true,
  "document_id": 1,
  "title": "Example",
  "chunks": 4
}
```

#### Ask a question

`POST /api/rag/query/`

Request body:

```json
{
  "query": "What does the document say about refunds?"
}
```

Returns:

```json
{
  "query_id": 1,
  "answer": "...",
  "sources": [
    {
      "document": "Policy",
      "content": "...",
      "similarity": 0.8123
    }
  ]
}
```

## Batch Ingestion

You can ingest files from disk with the management command:

```bash
python manage.py ingest_documents
python manage.py ingest_documents --directory documents
python manage.py ingest_documents --directory /absolute/path/to/files --reprocess
```

Supported file types are `.txt` and `.pdf`.

## Configuration Defaults

Current defaults from [django_rag/settings.py](/Users/reamonsumapig/Random/DjangoRag/django_rag/settings.py):

- `EMBEDDING_MODEL`: `sentence-transformers/all-MiniLM-L6-v2`
- `CHUNK_SIZE`: `1000`
- `CHUNK_OVERLAP`: `200`
- `MEDIA_ROOT`: `documents/`
- `TIME_ZONE`: `UTC`

## Important Implementation Notes

- Ingestion runs synchronously inside the upload request. Large files will make the upload request slow.
- Query answering makes a live external API call to Groq.
- The initial LangGraph rollout covers the query orchestration path only.
- Query results use the top 5 nearest chunks by L2 distance.
- The Docker setup is intended for local development and testing, not production hardening.
- Uploaded files are stored under the local `documents/` directory.
- `Document.content` stores the full extracted text, so database size grows with file volume.
- The current tests are lightweight and do not mock external embedding or LLM calls.

## Development Notes

- Main service logic lives in [rag_app/services/rag_service.py](/Users/reamonsumapig/Random/DjangoRag/rag_app/services/rag_service.py).
- UI templates live in [templates/rag_app](/Users/reamonsumapig/Random/DjangoRag/templates/rag_app).
- URL routing starts in [django_rag/urls.py](/Users/reamonsumapig/Random/DjangoRag/django_rag/urls.py) and delegates to [rag_app/urls.py](/Users/reamonsumapig/Random/DjangoRag/rag_app/urls.py).

For a deeper system breakdown, see [Architecture.md](/Users/reamonsumapig/Random/DjangoRag/Architecture.md).
