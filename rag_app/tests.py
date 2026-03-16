"""
Tests for Django RAG Application
Copyright (c) 2026 Reamon Sumapig - All Rights Reserved
"""
from django.test import TestCase
from django.test import Client
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile

from rag_app.models import Document, DocumentChunk, QueryHistory


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
