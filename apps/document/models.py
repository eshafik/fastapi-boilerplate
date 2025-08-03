from tortoise.models import Model
from tortoise import fields
from enum import Enum


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    PDF = "pdf"
    WEB = "web"
    TEXT = "text"


class Document(Model):
    id = fields.BigIntField(pk=True)
    title = fields.CharField(max_length=255)
    document_type = fields.CharEnumField(DocumentType)
    source_url = fields.TextField(null=True)  # For web/PDF URLs
    content = fields.TextField(null=True)  # For raw text
    metadata = fields.JSONField(default=dict)  # Additional metadata

    status = fields.CharEnumField(DocumentStatus, default=DocumentStatus.PENDING)
    error_message = fields.TextField(null=True)

    total_chunks = fields.IntField(default=0)
    total_tokens = fields.IntField(default=0)

    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "documents"
        indexes = [
            ("status", ),
            ("document_type", ),
        ]


class DocumentChunk(Model):
    id = fields.BigIntField(pk=True)
    document = fields.ForeignKeyField("models.Document", related_name="chunks", on_delete=fields.CASCADE)

    chunk_index = fields.IntField(null=True)  # Order within document
    content = fields.TextField(null=True)
    token_count = fields.IntField()

    # Metadata for retrieval
    page_number = fields.IntField(null=True)  # For PDFs
    section_title = fields.CharField(max_length=255, null=True)
    url = fields.TextField(null=True)  # For web chunks

    # Elasticsearch document ID (for updates/deletes)
    es_doc_id = fields.CharField(max_length=255, unique=True)

    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "document_chunks"
        indexes = [
            ("document_id", "chunk_index"),
            ("es_doc_id",),
        ]
