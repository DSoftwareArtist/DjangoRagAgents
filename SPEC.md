# Django RAG System Specification

## Project Overview
- **Project Name**: django_rag
- **Type**: Django Web Application with RAG System
- **Core Functionality**: Document ingestion, semantic search using embeddings, and question answering using Gemini LLM
- **Target Users**: Developers needing RAG capabilities in Django applications

## Technology Stack
- **Web Framework**: Django 4.2+
- **Database**: PostgreSQL 15+ with pgvector extension
- **Vector Store**: pgvector
- **LLM Framework**: LangChain
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **LLM**: Google Gemini

## Project Structure
```
DjangoRag/
├── django_rag/           # Main Django project
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── rag_app/              # Main RAG application
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── migrations/
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   └── management/
│       └── commands/
│           └── ingest_documents.py
├── documents/            # Document storage
├── requirements.txt
├── .env
├── manage.py
└── README.md
```

## Functionality Specification

### 1. Document Management
- Upload documents (TXT, PDF)
- Store document metadata in PostgreSQL
- Process and chunk documents for embedding

### 2. Vector Embeddings
- Generate embeddings using Sentence-Transformers
- Store embeddings in pgvector column
- Support semantic similarity search

### 3. RAG Query System
- Accept user queries
- Retrieve relevant documents using vector similarity
- Generate answers using Gemini via LangChain
- Display sources used

### 4. API Endpoints
- `POST /documents/upload/` - Upload document
- `GET /documents/` - List documents
- `POST /rag/query/` - Query RAG system
- `GET /rag/answer/<id>/` - Get previous query result

### 5. Web UI
- Simple Django templates for:
  - Document upload
  - Query interface
  - Results display

## Configuration
All settings via `.env` file:
- Database credentials
- Gemini API key
- Embedding model settings

## Acceptance Criteria
1. Django project runs without errors
2. PostgreSQL with pgvector is connected
3. Documents can be uploaded and stored
4. Embeddings are generated and stored in pgvector
5. Semantic search returns relevant results
6. Gemini LLM generates answers based on retrieved context
