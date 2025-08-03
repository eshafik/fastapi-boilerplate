from datetime import datetime

from pydantic import BaseModel, HttpUrl, validator, model_validator, ConfigDict
from typing import Optional, Dict, Any, List
from uuid import UUID

from apps.document.models import DocumentType, DocumentStatus


class DocumentCreateRequest(BaseModel):
    title: str
    document_type: DocumentType
    source_url: Optional[HttpUrl] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

    # Limits
    max_pages: Optional[int] = None  # For PDF/Web

    @model_validator(mode="after")
    def validate_required_fields(self) -> "DocumentCreateRequest":
        if self.document_type in {DocumentType.PDF, DocumentType.WEB} and not self.source_url:
            raise ValueError(f"`source_url` is required for {self.document_type} documents")
        if self.document_type == DocumentType.TEXT and not self.content:
            raise ValueError("`content` is required for text documents")
        return self


class DocumentResponse(BaseModel):
    id: int
    title: str
    document_type: DocumentType
    source_url: Optional[str]
    metadata: Dict[str, Any]
    status: DocumentStatus
    error_message: Optional[str]
    total_chunks: int
    total_tokens: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentUpdateRequest(BaseModel):
    title: Optional[str] = None
    source_url: Optional[HttpUrl] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    max_pages: Optional[int] = None