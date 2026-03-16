from django.urls import path
from . import views

urlpatterns = [
    path('', views.DocumentListView.as_view(), name='document_list'),
    path('documents/upload/', views.DocumentUploadView.as_view(), name='document_upload'),
    path('rag/query/', views.RAGQueryView.as_view(), name='rag_query'),
    path('rag/answer/<int:pk>/', views.QueryResultView.as_view(), name='query_result'),
    path('api/documents/upload/', views.api_upload_document, name='api_upload_document'),
    path('api/rag/query/', views.api_query, name='api_query'),
]
