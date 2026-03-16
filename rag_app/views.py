"""
Views - Django RAG Application
Copyright (c) 2026 Reamon Sumapig - All Rights Reserved
"""
from django.shortcuts import render, redirect
from django.views.generic import ListView, DetailView
from django.views import View
from django.contrib import messages
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings

from .models import Document, QueryHistory
from .services.rag_service import RAGService


class DocumentListView(ListView):
    model = Document
    template_name = 'rag_app/document_list.html'
    context_object_name = 'documents'
    ordering = ['-uploaded_at']


class DocumentUploadView(View):
    def get(self, request):
        return render(request, 'rag_app/document_upload.html')
    
    def post(self, request):
        if 'file' not in request.FILES:
            messages.error(request, 'No file uploaded')
            return redirect('document_upload')
        
        file = request.FILES['file']
        allowed_extensions = ['.txt', '.pdf']
        ext = file.name.lower()
        
        if not any(ext.endswith(e) for e in allowed_extensions):
            messages.error(request, 'Only .txt and .pdf files are allowed')
            return redirect('document_upload')
        
        document = Document.objects.create(
            title=request.POST.get('title', file.name),
            file=file,
        )
        
        try:
            rag_service = RAGService()
            rag_service.ingest_document(document)
            messages.success(request, 'Document uploaded and processed successfully')
        except Exception as e:
            messages.error(request, f'Error processing document: {str(e)}')
            document.delete()
        
        return redirect('document_list')


class RAGQueryView(View):
    def get(self, request):
        query_history = QueryHistory.objects.all().order_by('-created_at')[:10]
        return render(request, 'rag_app/rag_query.html', {'history': query_history})
    
    def post(self, request):
        query = request.POST.get('query', '').strip()
        
        if not query:
            messages.error(request, 'Please enter a query')
            return redirect('rag_query')
        
        try:
            rag_service = RAGService()
            answer, sources = rag_service.query(query)
            query_record = QueryHistory.objects.create(
                query=query,
                answer=answer,
                sources=sources,
            )
            
            return render(request, 'rag_app/rag_result.html', {
                'query': query,
                'answer': answer,
                'sources': sources,
                'query_id': query_record.id,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            messages.error(request, f'Error processing query: {str(e)}')
            return redirect('rag_query')


class QueryResultView(DetailView):
    model = QueryHistory
    template_name = 'rag_app/rag_result.html'
    context_object_name = 'query_record'
    pk_url_kwarg = 'pk'


def api_upload_document(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
    
    file = request.FILES['file']
    document = Document.objects.create(
        title=request.POST.get('title', file.name),
        file=file,
    )
    
    try:
        rag_service = RAGService()
        rag_service.ingest_document(document)
        return JsonResponse({
            'success': True,
            'document_id': document.id,
            'title': document.title,
            'chunks': document.chunks.count(),
        })
    except Exception as e:
        document.delete()
        return JsonResponse({'error': str(e)}, status=500)


def api_query(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    import json
    data = json.loads(request.body)
    query = data.get('query', '').strip()
    
    if not query:
        return JsonResponse({'error': 'Query is required'}, status=400)
    
    try:
        rag_service = RAGService()
        answer, sources = rag_service.query(query)
        
        query_record = QueryHistory.objects.create(
            query=query,
            answer=answer,
            sources=sources,
        )
        
        return JsonResponse({
            'query_id': query_record.id,
            'answer': answer,
            'sources': sources,
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
