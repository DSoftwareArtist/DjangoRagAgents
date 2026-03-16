"""
RAG Service - Retrieval Augmented Generation for Django
Copyright (c) 2026 Reamon Sumapig - All Rights Reserved
"""
import os
from typing import List, Tuple

from django.conf import settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from rag_app.models import Document, DocumentChunk


class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self):
        if self._model is None:
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self._model

    def embed_text(self, text: str) -> List[float]:
        model = self.get_model()
        embedding = model.encode(text)
        return embedding.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_text(query)


class TextProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )

    def process_file(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.pdf':
            return self._extract_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _extract_pdf(self, file_path: str) -> str:
        try:
            import PyPDF2
            text = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        except Exception as e:
            raise ValueError(f"Error extracting PDF: {e}")

    def chunk_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)


class RAGService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.text_processor = TextProcessor()
        self.client = OpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model_name = "llama-3.1-8b-instant"

    def ingest_document(self, document: Document) -> None:
        file_path = document.file.path
        content = self.text_processor.process_file(file_path)
        
        document.content = content
        document.save()

        chunks = self.text_processor.chunk_text(content)
        
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_service.embed_text(chunk)
            DocumentChunk.objects.create(
                document=document,
                content=chunk,
                chunk_index=i,
                embedding=embedding,
            )

        document.processed = True
        document.save()

    def retrieve_similar_chunks(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        from pgvector.django import L2Distance
        
        query_embedding = self.embedding_service.embed_query(query)
        
        chunks = (
            DocumentChunk.objects
            .annotate(distance=L2Distance('embedding', query_embedding))
            .order_by('distance')[:top_k]
        )
        
        results = []
        for chunk in chunks:
            distance = chunk.distance
            similarity = 1 / (1 + distance)
            results.append((chunk, similarity))
        
        return results

    def generate_answer(self, query: str, context_chunks: List[Tuple[DocumentChunk, float]]) -> str:
        context = "\n\n".join([chunk.content for chunk, _ in context_chunks])
        
        prompt = f"""Context: {context}

Question: {query}

Answer:"""
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return response.choices[0].message.content

    def query(self, query_text: str) -> Tuple[str, List[dict]]:
        similar_chunks = self.retrieve_similar_chunks(query_text)
        if not similar_chunks:
            return "No relevant documents found. Please upload and process documents first.", []
        answer = self.generate_answer(query_text, similar_chunks)
        sources = [
            {
                'document': chunk.document.title,
                'content': chunk.content[:200] + '...' if len(chunk.content) > 200 else chunk.content,
                'similarity': round(similarity, 4),
            }
            for chunk, similarity in similar_chunks
        ]
        
        return answer, sources
