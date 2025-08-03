import redis.asyncio as redis
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RedisSessionManager:
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl_hours: int = 24):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.ttl_seconds = ttl_hours * 3600
        self.max_history = 5  # Keep last 5 interactions per session

    def _session_history_key(self, session_id: str) -> str:
        return f"chat_history:{session_id}"

    def _session_meta_key(self, session_id: str) -> str:
        return f"chat_meta:{session_id}"

    def _chunks_key(self, session_id: str) -> str:
        return f"session_chunks:{session_id}"

    async def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get existing session or create new one"""
        if not session_id:
            session_id = str(uuid.uuid4())

        meta_key = self._session_meta_key(session_id)

        # Check if session exists, if not initialize it
        exists = await self.redis.exists(meta_key)
        if not exists:
            await self.redis.hset(meta_key, "created_at", datetime.utcnow().isoformat())
            await self.redis.expire(meta_key, self.ttl_seconds)

        return session_id

    async def add_interaction(self, session_id: str, user_query: str,
                              assistant_response: str, chunk_ids: List[str]):
        """Add user-assistant interaction to session history"""
        history_key = self._session_history_key(session_id)
        chunks_key = self._chunks_key(session_id)

        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_query": user_query,
            "assistant_response": assistant_response
        }

        # Add to conversation history (using list operations)
        await self.redis.lpush(history_key, json.dumps(interaction))
        await self.redis.ltrim(history_key, 0, self.max_history - 1)  # Keep only last N

        # Track used chunk IDs to avoid repetition
        if chunk_ids:
            await self.redis.sadd(chunks_key, *chunk_ids)

        # Reset TTL for all session keys
        await self.redis.expire(history_key, self.ttl_seconds)
        await self.redis.expire(chunks_key, self.ttl_seconds)

    async def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        history_key = self._session_history_key(session_id)

        # Check if history key exists first
        exists = await self.redis.exists(history_key)
        if not exists:
            return []

        history_json = await self.redis.lrange(history_key, 0, -1)

        history = []
        for item in reversed(history_json):  # Reverse to get chronological order
            try:
                history.append(json.loads(item))
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode history item: {item}")

        return history

    async def get_used_chunk_ids(self, session_id: str) -> List[str]:
        """Get chunk IDs used in this session to avoid repetition"""
        chunks_key = self._chunks_key(session_id)

        # Check if chunks key exists first
        exists = await self.redis.exists(chunks_key)
        if not exists:
            return []

        chunk_ids = await self.redis.smembers(chunks_key)
        return list(chunk_ids)

    async def close(self):
        await self.redis.close()