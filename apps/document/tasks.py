from celery import Celery
import asyncio
import logging
from config.celery import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def ingest_document_task(self, *args, **kwargs):
    """Synchronous Celery task that runs async logic"""
    try:
        print("kwargs", kwargs)
        document_id = kwargs.get('document_id')
        logger.info('document id', document_id)
        asyncio.run(_ingest_document_task(document_id))
    except Exception as e:
        logger.error(f"Retrying due to error: {str(e)}")
        raise self.retry(exc=e, countdown=60, max_retries=3)


async def _ingest_document_task(document_id: str):
    """Actual async ingestion logic"""
    from tortoise.transactions import in_transaction
    from tortoise import Tortoise
    from apps.document.ingestion_service import ingestion_service
    from apps.document.elasticsearch_client import es_client
    from apps.document.models import Document, DocumentChunk, DocumentStatus, DocumentType

    try:
        document = await Document.get(id=document_id)
        await document.update_from_dict({"status": DocumentStatus.PROCESSING})
        await document.save()

        logger.info(f"Starting ingestion for document {document_id}")

        if document.document_type == DocumentType.WEB:
            max_pages = document.metadata.get("max_pages", 1)
            print('max_pages', max_pages)
            documents = await ingestion_service.extract_web_content(document.source_url, max_pages)
            # print('documents', documents)
        elif document.document_type == DocumentType.PDF:
            max_pages = document.metadata.get("max_pages")
            documents = await ingestion_service.extract_pdf_content(document.source_url, max_pages)
            print('documents...', documents)
        elif document.document_type == DocumentType.TEXT:
            documents = await ingestion_service.process_text_content(document.content, document.title)
        else:
            raise ValueError(f"Unsupported document type: {document.document_type}")

        chunks = await ingestion_service.chunk_documents(documents)

        if not chunks:
            raise ValueError("No chunks were created from the document")

        chunk_texts = [chunk["content"] for chunk in chunks]
        print('==='*100)
        embeddings = await ingestion_service.generate_embeddings(chunk_texts)
        print('___'*50)

        db_chunks, es_chunks = await ingestion_service.prepare_chunks_for_indexing(
            chunks, document, embeddings
        )
        print('**'*100)

        await es_client.create_index_if_not_exists('test_index')

        async with in_transaction():
            await DocumentChunk.filter(document=document).delete()
            await es_client.delete_document_chunks(index_name='test_index', document_id=str(document.id))

            for db_chunk in db_chunks:
                await db_chunk.save()

            for es_chunk in es_chunks:
                await es_client.index_chunk("test_index", es_chunk)

            total_tokens = sum(chunk.token_count for chunk in db_chunks)
            await document.update_from_dict({
                "status": DocumentStatus.COMPLETED,
                "total_chunks": len(db_chunks),
                "total_tokens": total_tokens,
                "error_message": None
            })
            await document.save()

        logger.info(f"Successfully ingested document {document_id}: {len(db_chunks)} chunks, {total_tokens} tokens")

    except RuntimeError as e:
        logger.error(f"Failed to ingest document {document_id}: {str(e)}")
        try:
            document = await Document.get(id=document_id)
            await document.update_from_dict({
                "status": DocumentStatus.FAILED,
                "error_message": str(e)
            })
            await document.save()
        except:
            pass
        raise
    finally:
        await Tortoise.close_connections()
