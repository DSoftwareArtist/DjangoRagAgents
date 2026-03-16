# Django RAG System

A Django-based Retrieval-Augmented Generation (RAG) system using PostgreSQL with pgvector, LangChain, Sentence-Transformers, and Groq (Llama).

## Tech Stack

- **Django 4.2** - Web framework
- **PostgreSQL + pgvector** - Vector database
- **LangChain** - LLM orchestration
- **Sentence-Transformers** - Embeddings (all-MiniLM-L6-v2)
- **Groq** - LLM for answer generation (Llama 3.1 8B)
- **PyPDF2** - PDF text extraction

## Setup Instructions

### 1. Prerequisites

- Python 3.9+
- PostgreSQL 15+ with pgvector extension
- Groq API key (free)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Database Setup

**Option A: Using Docker**
```bash
docker run -d --name postgres-pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

**Option B: Local PostgreSQL**
1. Install PostgreSQL
2. Create database: `CREATE DATABASE rag_db;`
3. Enable pgvector: `CREATE EXTENSION vector;`

### 4. Configure Environment

Edit `.env` file:
```env
DB_NAME=rag_db
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_django_secret_key
```

**Get Groq API Key (free):**
1. Go to https://console.groq.com/keys
2. Create a new API key
3. Copy it to your `.env` file

### 5. Run Migrations

```bash
python manage.py migrate
```

### 6. Run Development Server

```bash
python manage.py runserver
```

### 7. Access the Application

- Web UI: http://localhost:8000
- Admin: http://localhost:8000/admin

## Usage

### Upload Documents
1. Go to `/` or click "Documents"
2. Click "Upload Document"
3. Upload a .txt or .pdf file
4. The document will be automatically chunked and embedded

### Query the RAG System
1. Go to `/rag/query/`
2. Enter your question
3. The system will:
   - Search for relevant document chunks using vector similarity
   - Generate an answer using Groq (Llama 3.1)

### API Endpoints

- `POST /api/documents/upload/` - Upload document
- `POST /api/rag/query/` - Query RAG system

```bash
# Upload document
curl -X POST http://localhost:8000/api/documents/upload/ \
  -F "title=My Document" \
  -F "file=@document.pdf"

# Query
curl -X POST http://localhost:8000/api/rag/query/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?"}'
```

## Management Commands

```bash
# Ingest documents from directory
python manage.py ingest_documents --directory=documents
```

## Running Tests

```bash
# Run all tests
python manage.py test

# Run with verbose output
python manage.py test rag_app --verbosity=2
```

## GitHub Actions

The project includes CI/CD workflows in `.github/workflows/`:
- **Django Tests**: Runs migrations and unit tests on push/PR
- **Lint**: Code quality checks with flake8

## Notes

- Groq provides free tier with Llama 3.1 8B model (very fast)
- Embeddings are stored in PostgreSQL using pgvector
- Documents are automatically chunked for optimal retrieval
