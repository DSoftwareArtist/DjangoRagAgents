import os
from django.core.management.base import BaseCommand
from django.conf import settings

from rag_app.models import Document
from rag_app.services.rag_service import RAGService


class Command(BaseCommand):
    help = 'Ingest documents from the documents directory'

    def add_arguments(self, parser):
        parser.add_argument(
            '--directory',
            type=str,
            default='documents',
            help='Directory containing documents to ingest',
        )
        parser.add_argument(
            '--reprocess',
            action='store_true',
            help='Reprocess already processed documents',
        )

    def handle(self, *args, **options):
        dir_path = options['directory']
        reprocess = options['reprocess']
        
        if not os.path.isabs(dir_path):
            dir_path = os.path.join(settings.BASE_DIR, dir_path)
        
        if not os.path.exists(dir_path):
            self.stdout.write(self.style.ERROR(f'Directory {dir_path} does not exist'))
            return
        
        allowed_extensions = ['.txt', '.pdf']
        files = [
            f for f in os.listdir(dir_path)
            if os.path.splitext(f.lower())[1] in allowed_extensions
        ]
        
        if not files:
            self.stdout.write(self.style.WARNING(f'No files found in {dir_path}'))
            return
        
        rag_service = RAGService()
        
        for filename in files:
            file_path = os.path.join(dir_path, filename)
            
            title = os.path.splitext(filename)[0]
            
            doc, created = Document.objects.get_or_create(
                title=title,
                defaults={'file': os.path.relpath(file_path, settings.MEDIA_ROOT)}
            )
            
            if not created and doc.processed and not reprocess:
                self.stdout.write(self.style.WARNING(f'Skipped (already processed): {filename}'))
                continue
            
            try:
                if not doc.file:
                    doc.file = os.path.relpath(file_path, settings.MEDIA_ROOT)
                    doc.save()
                
                rag_service.ingest_document(doc)
                self.stdout.write(self.style.SUCCESS(f'Processed: {filename}'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error processing {filename}: {str(e)}'))
        
        self.stdout.write(self.style.SUCCESS(f'Done! Processed {len(files)} files'))
