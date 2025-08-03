from typing import Optional

# document/views.py
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

from apps.document.elasticsearch_client import es_client
from apps.document.models import Document, DocumentStatus, DocumentType, DocumentChunk
from apps.document.schemas import DocumentCreateRequest, DocumentResponse, DocumentUpdateRequest
from apps.document.tasks import ingest_document_task, _ingest_document_task


async def create_document(request: DocumentCreateRequest):
    """Create a new document and start ingestion"""
    try:
        # Prepare metadata
        metadata = request.metadata or {}
        print('max_pages', request.max_pages)
        if request.max_pages:
            metadata["max_pages"] = request.max_pages

        # Create document record
        document = Document(
            title=request.title,
            document_type=request.document_type,
            source_url=str(request.source_url) if request.source_url else None,
            content=request.content,
            metadata=metadata,
            status=DocumentStatus.PENDING
        )
        await document.save()
        await _ingest_document_task(document_id=document.pk)

        print('document id', document.id)
        # Start ingestion task
        # ingest_document_task.delay(document_id=str(document.id))

        data = DocumentResponse.from_orm(document)
        print('data', data.dict(), 'type', type(data))
        return {'data': jsonable_encoder(data)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_document(document_id: str):
    """Get document by ID"""
    try:
        document = await Document.get(id=document_id)
        return DocumentResponse.from_orm(document)
    except Exception:
        raise HTTPException(status_code=404, detail="Document not found")


async def list_documents(
        status: Optional[DocumentStatus] = None,
        document_type: Optional[DocumentType] = None,
        limit: int = 100,
        offset: int = 0
):
    """List documents with optional filtering"""
    query = Document.all()
    if status:
        query = query.filter(status=status)
    if document_type:
        query = query.filter(document_type=document_type)

    documents = await query.offset(offset).limit(limit).order_by("-created_at")
    return [DocumentResponse.from_orm(doc) for doc in documents]


async def update_document(document_id: str, request: DocumentUpdateRequest):
    """Update document and re-ingest"""
    try:
        document = await Document.get(id=document_id)

        # Update fields
        update_data = {}
        if request.title is not None:
            update_data["title"] = request.title
        if request.source_url is not None:
            update_data["source_url"] = str(request.source_url)
        if request.content is not None:
            update_data["content"] = request.content
        if request.metadata is not None:
            update_data["metadata"] = request.metadata

        # Handle max_pages
        if request.max_pages is not None:
            metadata = document.metadata or {}
            metadata["max_pages"] = request.max_pages
            update_data["metadata"] = metadata

        # Reset status for re-ingestion
        update_data["status"] = DocumentStatus.PENDING
        update_data["error_message"] = None

        await document.update_from_dict(update_data)
        await document.save()

        # Start re-ingestion
        ingest_document_task.delay(str(document.id))

        return DocumentResponse.from_orm(document)

    except Exception as e:
        if "DoesNotExist" in str(e):
            raise HTTPException(status_code=404, detail="Document not found")
        raise HTTPException(status_code=500, detail=str(e))


async def delete_document(document_id: str):
    """Delete document and all its chunks"""
    try:
        document = await Document.get(id=document_id)

        # Delete from Elasticsearch
        await es_client.delete_document_chunks(document_id)

        # Delete from database (cascades to chunks)
        await document.delete()

        return {"message": "Document deleted successfully"}

    except Exception as e:
        if "DoesNotExist" in str(e):
            raise HTTPException(status_code=404, detail="Document not found")
        raise HTTPException(status_code=500, detail=str(e))


async def get_document_chunks(document_id: str, limit: int = 100, offset: int = 0):
    """Get chunks for a document"""
    try:
        document = await Document.get(id=document_id)
        chunks = await DocumentChunk.filter(document=document).offset(offset).limit(limit).order_by("chunk_index")

        return {
            "document_id": document_id,
            "total_chunks": await DocumentChunk.filter(document=document).count(),
            "chunks": [
                {
                    "id": str(chunk.id),
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "url": chunk.url,
                }
                for chunk in chunks
            ]
        }

    except Exception as e:
        if "DoesNotExist" in str(e):
            raise HTTPException(status_code=404, detail="Document not found")
        raise HTTPException(status_code=500, detail=str(e))