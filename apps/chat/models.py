# models/chat.py
from tortoise.models import Model
from tortoise import fields
from typing import Optional, List, Dict, Any
from datetime import datetime
import json


class ChatSession(Model):
    """Chat session model to track conversations"""
    id = fields.UUIDField(pk=True)
    user_id = fields.CharField(max_length=255, index=True)
    session_name = fields.CharField(max_length=255, null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    is_active = fields.BooleanField(default=True)

    # Store session configuration
    config = fields.JSONField(default=dict)  # max_messages, temperature, etc.

    class Meta:
        table = "chat_sessions"


class ChatMessage(Model):
    """Individual chat messages"""
    id = fields.UUIDField(pk=True)
    session = fields.ForeignKeyField("models.ChatSession", related_name="messages")

    role = fields.CharField(max_length=20)  # 'user', 'assistant', 'system'
    content = fields.TextField()

    # Store used chunks for this message
    used_chunk_ids = fields.JSONField(default=list)

    # Store retrieval metadata
    retrieval_metadata = fields.JSONField(default=dict)  # scores, query, etc.

    created_at = fields.DatetimeField(auto_now_add=True)
    token_count = fields.IntField(null=True)

    class Meta:
        table = "chat_messages"
        ordering = ["created_at"]


class SessionChunkMemory(Model):
    """Track chunks used across the entire session to avoid repetition"""
    id = fields.UUIDField(pk=True)
    session = fields.ForeignKeyField("models.ChatSession", related_name="chunk_memory")
    chunk_id = fields.CharField(max_length=255)

    # Metadata about when and how this chunk was used
    first_used_at = fields.DatetimeField(auto_now_add=True)
    usage_count = fields.IntField(default=1)
    last_used_at = fields.DatetimeField(auto_now=True)

    # Relevance score when first retrieved
    relevance_score = fields.FloatField(null=True)

    class Meta:
        table = "session_chunk_memory"
        unique_together = ("session", "chunk_id")
