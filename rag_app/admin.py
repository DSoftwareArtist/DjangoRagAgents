from django.contrib import admin
from .models import Document, DocumentChunk, QueryHistory


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['title', 'uploaded_at', 'processed']
    list_filter = ['processed', 'uploaded_at']
    search_fields = ['title', 'content']


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ['document', 'chunk_index', 'created_at']
    list_filter = ['document']


@admin.register(QueryHistory)
class QueryHistoryAdmin(admin.ModelAdmin):
    list_display = ['query', 'created_at']
    list_filter = ['created_at']
    search_fields = ['query', 'answer']
