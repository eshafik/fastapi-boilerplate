from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ChatRequest(BaseModel):
    query: str = Field(..., description="User query", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation continuity")
    top_k: int = Field(default=5, description="Number of chunks to retrieve", ge=1, le=20)
    bm25_weight: float = Field(default=1.0, description="Weight for BM25 scoring", ge=0.0, le=2.0)
    vector_weight: float = Field(default=1.0, description="Weight for vector scoring", ge=0.0, le=2.0)


class UsedChunk(BaseModel):
    chunk_id: str
    score: float
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    response: str
    session_id: str
    rag_time_ms: int
    llm_time_ms: int
    tokens_input: int
    tokens_output: int
    used_chunks: List[UsedChunk]
