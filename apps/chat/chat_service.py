# services/chat_service.py
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID, uuid4
import logging
from datetime import datetime, timedelta

from openai import AsyncOpenAI
from tortoise.transactions import in_transaction

from apps.chat.models import ChatSession, ChatMessage, SessionChunkMemory
from apps.document.elasticsearch_client import es_client
from config.settings import OPENAI_API_KEY, ELASTICSEARCH_INDEX_NAME

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-4o-mini"  # or gpt-4o for higher quality

        # Configuration
        self.max_conversation_history = 10  # Last N messages to include
        self.chunk_memory_decay_days = 7  # How long to remember used chunks
        self.max_chunk_reuse_count = 3  # Max times to reuse same chunk

    async def create_or_get_session(
            self,
            user_id: str,
            session_id: Optional[UUID] = None,
            session_name: Optional[str] = None
    ) -> ChatSession:
        """Create new session or retrieve existing one"""

        if session_id:
            session = await ChatSession.get_or_none(id=session_id, user_id=user_id, is_active=True)
            if session:
                return session

        # Create new session
        session = await ChatSession.create(
            user_id=user_id,
            session_name=session_name or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            config={
                "max_chunks": 5,
                "temperature": 0.7,
                "max_tokens": 1000,
                "created_via": "api"
            }
        )

        logger.info(f"Created new chat session {session.id} for user {user_id}")
        return session

    async def get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text"""
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    async def get_conversation_context(self, session: ChatSession, limit: int = None) -> List[Dict[str, str]]:
        """Get recent conversation history for context"""
        limit = limit or self.max_conversation_history

        messages = await ChatMessage.filter(
            session=session
        ).order_by('-created_at').limit(limit).values(
            'role', 'content', 'created_at'
        )

        # Reverse to get chronological order
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in reversed(messages)
        ]

    async def get_session_chunk_memory(self, session: ChatSession) -> List[str]:
        """Get list of chunk IDs already used in this session"""

        # Clean up old memory first
        cutoff_date = datetime.now() - timedelta(days=self.chunk_memory_decay_days)
        await SessionChunkMemory.filter(
            session=session,
            last_used_at__lt=cutoff_date
        ).delete()

        # Get current memory
        memory = await SessionChunkMemory.filter(
            session=session,
            usage_count__lt=self.max_chunk_reuse_count
        ).values_list('chunk_id', flat=True)

        return list(memory)

    async def hybrid_retrieve(
            self,
            query: str,
            embedding: List[float],
            exclude_chunk_ids: Optional[List[str]] = None,
            max_chunks: int = 5
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Perform hybrid retrieval and return results with metadata"""

        start_time = time.time()

        try:
            results = await es_client.hybrid_search(
                index_name=ELASTICSEARCH_INDEX_NAME,
                query=query,
                embedding=embedding,
                limit=max_chunks * 2,  # Get more to have options after filtering
                exclude_chunk_ids=exclude_chunk_ids or []
            )

            # Take top results after exclusion
            final_results = results[:max_chunks]

            retrieval_metadata = {
                "query": query,
                "total_found": len(results),
                "excluded_chunks": len(exclude_chunk_ids or []),
                "retrieval_time_ms": int((time.time() - start_time) * 1000),
                "search_type": "hybrid_bm25_vector"
            }

            return final_results, retrieval_metadata

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return [], {"error": str(e), "retrieval_time_ms": 0}

    async def update_chunk_memory(
            self,
            session: ChatSession,
            chunk_ids: List[str],
            scores: List[float]
    ):
        """Update session chunk memory with used chunks"""

        for chunk_id, score in zip(chunk_ids, scores):
            memory, created = await SessionChunkMemory.get_or_create(
                session=session,
                chunk_id=chunk_id,
                defaults={
                    "relevance_score": score,
                    "usage_count": 1
                }
            )

            if not created:
                memory.usage_count += 1
                memory.last_used_at = datetime.now()
                memory.relevance_score = max(memory.relevance_score or 0, score)
                await memory.save()

    def build_rag_prompt(
            self,
            user_query: str,
            retrieved_chunks: List[Dict[str, Any]],
            conversation_history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Build optimized prompt for RAG with conversation context"""

        # System message
        system_message = {
            "role": "system",
            "content": """You are a helpful AI assistant with access to a knowledge base. Use the provided context to answer questions accurately and comprehensively.

INSTRUCTIONS:
- Answer based on the provided context when relevant
- If the context doesn't contain enough information, say so clearly
- Maintain conversation flow and reference previous messages when appropriate
- Be concise but thorough
- If asked about sources, mention that you're drawing from the user's knowledge base

CONTEXT INFORMATION:"""
        }

        # Add retrieved chunks to system message
        if retrieved_chunks:
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                metadata = chunk.get('metadata', {})
                source_info = f"[Source {i}"
                if metadata.get('document_title'):
                    source_info += f": {metadata['document_title']}"
                if metadata.get('page_number'):
                    source_info += f", Page {metadata['page_number']}"
                source_info += "]"

                context_parts.append(f"{source_info}\n{chunk['content']}")

            system_message["content"] += "\n\n" + "\n\n".join(context_parts)

        # Build message list
        messages = [system_message]

        # Add conversation history (recent messages)
        if conversation_history:
            messages.extend(conversation_history)

        # Add current user query
        messages.append({"role": "user", "content": user_query})

        return messages

    async def generate_response(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: int = 1000
    ) -> Tuple[str, Dict[str, int]]:
        """Generate response using OpenAI"""

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )

            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

            return response.choices[0].message.content, token_usage

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    async def process_chat_message(
            self,
            user_id: str,
            message: str,
            session_id: Optional[UUID] = None,
            max_chunks: int = 5,
            temperature: float = 0.7,
            max_tokens: int = 1000,
            include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Main chat processing pipeline"""

        start_time = time.time()

        async with in_transaction():
            # Get or create session
            session = await self.create_or_get_session(user_id, session_id)

            # Get embedding for the query
            embedding = await self.get_embedding(message)

            # Get conversation context
            conversation_history = await self.get_conversation_context(session)

            # Get chunk memory to avoid repetition
            used_chunk_ids = await self.get_session_chunk_memory(session)

            # Perform hybrid retrieval
            retrieved_chunks, retrieval_metadata = await self.hybrid_retrieve(
                query=message,
                embedding=embedding,
                exclude_chunk_ids=used_chunk_ids,
                max_chunks=max_chunks
            )

            # Build RAG prompt
            prompt_messages = self.build_rag_prompt(
                user_query=message,
                retrieved_chunks=retrieved_chunks,
                conversation_history=conversation_history
            )

            # Generate response
            response_text, token_usage = await self.generate_response(
                messages=prompt_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Save user message
            user_message = await ChatMessage.create(
                session=session,
                role="user",
                content=message,
                used_chunk_ids=[],
                retrieval_metadata=retrieval_metadata
            )

            # Save assistant response
            chunk_ids = [chunk['chunk_id'] for chunk in retrieved_chunks]
            chunk_scores = [chunk['score'] for chunk in retrieved_chunks]

            assistant_message = await ChatMessage.create(
                session=session,
                role="assistant",
                content=response_text,
                used_chunk_ids=chunk_ids,
                retrieval_metadata={
                    **retrieval_metadata,
                    "token_usage": token_usage
                },
                token_count=token_usage.get("total_tokens", 0)
            )

            # Update chunk memory
            if chunk_ids:
                await self.update_chunk_memory(session, chunk_ids, chunk_scores)

            # Update session
            session.updated_at = datetime.now()
            await session.save()

            # Prepare response
            processing_time = int((time.time() - start_time) * 1000)

            retrieved_chunks_formatted = []
            for chunk in retrieved_chunks:
                retrieved_chunks_formatted.append({
                    "chunk_id": chunk['chunk_id'],
                    "content": chunk['content'],
                    "score": chunk['score'],
                    "metadata": chunk['metadata'] if include_metadata else {},
                    "source_type": "fresh"
                })

            message_count = await ChatMessage.filter(session=session).count()

            return {
                "message_id": assistant_message.id,
                "session_id": session.id,
                "response": response_text,
                "retrieved_chunks": retrieved_chunks_formatted,
                "total_chunks_available": len(retrieved_chunks),
                "token_usage": token_usage,
                "processing_time_ms": processing_time,
                "message_count": message_count,
                "created_at": assistant_message.created_at
            }


# Global service instance
chat_service = ChatService()