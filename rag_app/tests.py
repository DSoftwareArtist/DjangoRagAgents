"""
Tests for Django RAG Application
Copyright (c) 2026 Reamon Sumapig - All Rights Reserved
"""
from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase
from django.test import Client
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile

from rag_app.models import Document, DocumentChunk, QueryHistory
from rag_app.services.rag_service import RAGService


class DocumentModelTest(TestCase):
    def test_create_document(self):
        doc = Document.objects.create(
            title="Test Document",
            content="Test content",
            processed=False
        )
        self.assertEqual(doc.title, "Test Document")
        self.assertFalse(doc.processed)
        self.assertIsNotNone(doc.uploaded_at)

    def test_document_str(self):
        doc = Document(title="Test")
        self.assertEqual(str(doc), "Test")


class DocumentChunkModelTest(TestCase):
    def setUp(self):
        self.doc = Document.objects.create(
            title="Test Doc",
            content="Test content"
        )

    def test_create_chunk(self):
        chunk = DocumentChunk.objects.create(
            document=self.doc,
            content="Test chunk",
            chunk_index=0,
            embedding=[0.1] * 384
        )
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(chunk.document.title, "Test Doc")


class QueryHistoryModelTest(TestCase):
    def test_create_query(self):
        query = QueryHistory.objects.create(
            query="What is this?",
            answer="This is a test.",
            sources=[]
        )
        self.assertEqual(query.query, "What is this?")
        self.assertEqual(query.answer, "This is a test.")


class DocumentUploadViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_upload_view_get(self):
        response = self.client.get('/documents/upload/')
        self.assertEqual(response.status_code, 200)

    def test_upload_view_post(self):
        txt_file = SimpleUploadedFile(
            "test.txt",
            b"Test content for document",
            content_type="text/plain"
        )
        response = self.client.post('/documents/upload/', {
            'title': 'Test Document',
            'file': txt_file
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Document.objects.filter(title='Test Document').exists())


class RAGQueryViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_query_view_get(self):
        response = self.client.get('/rag/query/')
        self.assertEqual(response.status_code, 200)

    def test_query_view_post_empty(self):
        response = self.client.post('/rag/query/', {
            'query': ''
        })
        self.assertEqual(response.status_code, 302)


class DocumentListViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        Document.objects.create(title="Doc 1", content="Content 1")
        Document.objects.create(title="Doc 2", content="Content 2")

    def test_list_view(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Doc 1")
        self.assertContains(response, "Doc 2")


class RAGServiceQueryTest(TestCase):
    def setUp(self):
        self.service = RAGService()

    def test_query_returns_empty_result_when_no_chunks_match(self):
        with patch.object(self.service, 'retrieve_similar_chunks', return_value=[]), patch.object(
            self.service, 'generate_answer'
        ) as mock_generate_answer:
            answer, sources = self.service.query("Missing context?")

        self.assertEqual(
            answer,
            "No relevant documents found. Please upload and process documents first.",
        )
        self.assertEqual(sources, [])
        mock_generate_answer.assert_not_called()

    def test_query_formats_sources_from_matching_chunks(self):
        document = Document(title="Handbook")
        chunk = DocumentChunk(
            document=document,
            content="A" * 220,
            chunk_index=0,
            embedding=[0.1] * 384,
        )
        similar_chunks = [(chunk, 0.81234)]

        with patch.object(self.service, 'retrieve_similar_chunks', return_value=similar_chunks), patch.object(
            self.service, 'generate_answer', return_value="Generated answer"
        ):
            answer, sources = self.service.query("What is in the handbook?")

        self.assertEqual(answer, "Generated answer")
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]['document'], "Handbook")
        self.assertEqual(sources[0]['similarity'], 0.8123)
        self.assertTrue(sources[0]['content'].endswith('...'))

    def test_generate_answer_uses_document_only_guardrail_prompt(self):
        document = Document(title="Policy")
        chunk = DocumentChunk(
            document=document,
            content="Refunds are allowed within 30 days of purchase.",
            chunk_index=0,
            embedding=[0.1] * 384,
        )

        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Refunds are allowed within 30 days of purchase."))]
        )

        with patch.object(self.service.client.chat.completions, 'create', return_value=mock_response) as mock_create:
            answer = self.service.generate_answer("What is the refund policy?", [(chunk, 0.91)])

        self.assertEqual(answer, "Refunds are allowed within 30 days of purchase.")
        _, kwargs = mock_create.call_args
        messages = kwargs['messages']
        self.assertEqual(messages[0]['role'], 'system')
        self.assertIn("Answer using only the provided document context.", messages[0]['content'])
        self.assertIn(self.service.document_only_fallback, messages[0]['content'])
        self.assertIn("Document context:", messages[1]['content'])
        self.assertIn("Refunds are allowed within 30 days of purchase.", messages[1]['content'])
