from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID


class ChatMessageRequestV1(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: Optional[UUID] = None
    user_id: str = Field(..., min_length=1)

    # Retrieval configuration
    max_chunks: int = Field(default=5, ge=1, le=20)
    include_metadata: bool = Field(default=True)

    # Generation configuration
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=100, le=4000)


class RetrievedChunkV1(BaseModel):
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source_type: str  # 'fresh', 'memory', 'hybrid'


class ChatMessageResponseV1(BaseModel):
    message_id: UUID
    session_id: UUID
    response: str

    # Retrieval information
    retrieved_chunks: List[RetrievedChunkV1]
    total_chunks_available: int

    # Generation metadata
    token_usage: Dict[str, int]
    processing_time_ms: int

    # Session info
    message_count: int
    created_at: datetime


class ChatSessionInfo(BaseModel):
    session_id: UUID
    user_id: str
    session_name: Optional[str]
    message_count: int
    created_at: datetime
    updated_at: datetime
    is_active: bool


class ChatHistoryResponse(BaseModel):
    session: ChatSessionInfo
    messages: List[Dict[str, Any]]
    total_messages: int
    page: int
    page_size: int


class RetrievedChunk(BaseModel):
    """Model for retrieved chunks in responses"""
    chunk_id: str
    content: str
    score: float
    adjusted_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        schema_extra = {
            "example": {
                "chunk_id": "doc1_chunk_1",
                "content": "This is relevant content...",
                "score": 0.85,
                "adjusted_score": 0.87,
                "metadata": {
                    "document_title": "User Manual",
                    "page_number": 42,
                    "section_title": "Getting Started"
                }
            }
        }


class ProcessingStep(BaseModel):
    """Model for processing step timing"""
    step: str
    duration_ms: int
    cache_hit: Optional[bool] = None
    prompt_tokens: Optional[int] = None
    available_tokens: Optional[int] = None


class TokenUsage(BaseModel):
    """Model for token usage information"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class PerformanceMetrics(BaseModel):
    """Model for performance metrics"""
    cache_hits: int = 0
    total_steps: int = 0
    retrieval_efficiency: float = 0.0
    token_efficiency: float = 0.0


class ChatResponse(BaseModel):
    """Main chat response model"""
    message_id: UUID
    session_id: UUID
    response: str

    # Retrieval information
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    total_chunks_available: int = 0
    chunks_selected: int = 0

    # Performance and usage
    token_usage: TokenUsage
    processing_time_ms: int
    processing_steps: List[ProcessingStep] = Field(default_factory=list)

    # Session information
    message_count: int
    created_at: datetime

    # Advanced metrics
    performance_metrics: PerformanceMetrics

    class Config:
        schema_extra = {
            "example": {
                "message_id": "123e4567-e89b-12d3-a456-426614174000",
                "session_id": "123e4567-e89b-12d3-a456-426614174001",
                "response": "Based on the documentation, here's how to get started...",
                "retrieved_chunks": [
                    {
                        "chunk_id": "doc1_chunk_1",
                        "content": "Getting started is easy...",
                        "score": 0.85,
                        "metadata": {"document_title": "User Guide"}
                    }
                ],
                "total_chunks_available": 15,
                "chunks_selected": 3,
                "token_usage": {
                    "prompt_tokens": 450,
                    "completion_tokens": 125,
                    "total_tokens": 575
                },
                "processing_time_ms": 1250,
                "message_count": 5,
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class ChatSessionResponse(BaseModel):
    """Chat session response model"""
    id: UUID
    user_id: str
    session_name: str
    config: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "user_id": "user123",
                "session_name": "Technical Support Chat",
                "config": {
                    "max_chunks": 5,
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                "is_active": True,
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "message_count": 8
            }
        }


class ChatMessageRequest(BaseModel):
    """Legacy chat message request for backward compatibility"""
    user_id: str
    message: str
    session_id: Optional[UUID] = None
    max_chunks: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=100, le=4000)
    include_metadata: bool = Field(default=True)


class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        schema_extra = {
            "example": {
                "error": "Session not found",
                "error_code": "SESSION_NOT_FOUND",
                "details": {"session_id": "123e4567-e89b-12d3-a456-426614174000"},
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class StreamingChatResponse(BaseModel):
    """Streaming response model for SSE"""
    type: str  # metadata, content, complete, error
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class ConversationQualityMetrics(BaseModel):
    """Model for conversation quality analysis"""
    session_id: str
    conversation_length: int
    user_messages: int
    assistant_messages: int

    token_efficiency: Dict[str, float]
    chunk_usage: Dict[str, float]
    performance: Dict[str, float]
    overall_quality_score: float

    class Config:
        schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "conversation_length": 16,
                "user_messages": 8,
                "assistant_messages": 8,
                "token_efficiency": {
                    "total_tokens": 4500,
                    "avg_tokens_per_response": 562.5,
                    "token_efficiency_score": 0.85
                },
                "chunk_usage": {
                    "unique_chunks_used": 25,
                    "total_chunk_references": 32,
                    "chunk_reuse_rate": 0.22,
                    "avg_chunks_per_response": 4.0
                },
                "performance": {
                    "avg_response_time_ms": 1450,
                    "response_time_score": 0.71
                },
                "overall_quality_score": 0.79
            }
        }


class ChatRequest(BaseModel):
    """Enhanced chat request model with all configuration options"""
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    session_id: Optional[UUID] = Field(None, description="Existing session ID")
    session_name: Optional[str] = Field(None, max_length=255, description="Custom session name")

    # Retrieval configuration
    max_chunks: int = Field(default=5, ge=1, le=20, description="Maximum chunks to retrieve")
    include_metadata: bool = Field(default=True, description="Include retrieval metadata in response")

    # Generation configuration
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Response creativity")
    max_tokens: int = Field(default=1000, ge=100, le=4000, description="Maximum response tokens")

    # Advanced options
    user_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User preferences")
    stream: bool = Field(default=False, description="Enable streaming response")


class ChatHistoryRequest(BaseModel):
    """Request model for chat history"""
    session_id: UUID
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    include_metadata: bool = Field(default=False)


class SessionUpdateRequest(BaseModel):
    """Request model for updating session"""
    session_name: Optional[str] = Field(None, max_length=255)
    is_active: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None