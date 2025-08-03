# services/enhanced_chat_service.py
import asyncio
import time
import json
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from uuid import UUID, uuid4
import logging
from datetime import datetime, timedelta

from openai import AsyncOpenAI
from tortoise.transactions import in_transaction

from models.chat import ChatSession, ChatMessage, SessionChunkMemory
from elasticsearch_client import es_client
from config.settings import OPENAI_API_KEY, ELASTICSEARCH_INDEX_NAME
from utils.response_formatter import response_formatter, token_counter

logger = logging.getLogger(__name__)


class EnhancedChatService:
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-4o-mini"

        # Advanced configuration
        self.max_conversation_history = 10
        self.chunk_memory_decay_days = 7
        self.max_chunk_reuse_count = 3

        # Performance settings
        self.embedding_cache = {}  # Simple in-memory cache
        self.cache_ttl = 3600  # 1 hour

        # Quality settings
        self.min_chunk_score_threshold = 0.1
        self.semantic_similarity_weight = 0.7
        self.keyword_match_weight = 0.3

    async def get_cached_embedding(self, text: str) -> List[float]:
        """Get embedding with caching"""

        # Simple cache key
        cache_key = hash(text)
        current_time = time.time()

        # Check cache
        if cache_key in self.embedding_cache:
            cached_data = self.embedding_cache[cache_key]
            if current_time - cached_data['timestamp'] < self.cache_ttl:
                return cached_data['embedding']

        # Get new embedding
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding

            # Cache it
            self.embedding_cache[cache_key] = {
                'embedding': embedding,
                'timestamp': current_time
            }

            return embedding

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    async def intelligent_chunk_selection(
            self,
            query: str,
            raw_chunks: List[Dict[str, Any]],
            conversation_history: List[Dict[str, str]],
            max_chunks: int = 5
    ) -> List[Dict[str, Any]]:
        """Intelligently select the most relevant chunks based on context"""

        if not raw_chunks:
            return []

        # Score chunks based on multiple factors
        scored_chunks = []

        for chunk in raw_chunks:
            score = chunk['score']

            # Boost score based on recency of document
            metadata = chunk.get('metadata', {})
            if 'created_at' in metadata:
                # Newer documents get slight boost
                doc_age_days = (datetime.now() - datetime.fromisoformat(metadata['created_at'])).days
                recency_boost = max(0, 1 - (doc_age_days / 365)) * 0.1
                score += recency_boost

            # Boost score if chunk relates to recent conversation
            if conversation_history:
                recent_context = " ".join([msg['content'] for msg in conversation_history[-3:]])
                # Simple keyword overlap check
                chunk_words = set(chunk['content'].lower().split())
                context_words = set(recent_context.lower().split())
                overlap = len(chunk_words.intersection(context_words))
                if overlap > 0:
                    score += min(overlap * 0.05, 0.2)  # Cap the boost

            # Apply minimum threshold
            if score >= self.min_chunk_score_threshold:
                scored_chunks.append({
                    **chunk,
                    'adjusted_score': score
                })

        # Sort by adjusted score and take top chunks
        scored_chunks.sort(key=lambda x: x['adjusted_score'], reverse=True)
        return scored_chunks[:max_chunks]

    async def build_dynamic_prompt(
            self,
            user_query: str,
            retrieved_chunks: List[Dict[str, Any]],
            conversation_history: List[Dict[str, str]],
            user_preferences: Dict[str, Any] = None
    ) -> List[Dict[str, str]]:
        """Build dynamic, context-aware prompts"""

        user_preferences = user_preferences or {}

        # Adaptive system message based on query type and context
        system_parts = [
            "You are a knowledgeable AI assistant with access to a curated knowledge base.",
            "Provide accurate, helpful, and contextually relevant responses."
        ]

        # Analyze query type
        query_lower = user_query.lower()

        if any(word in query_lower for word in ['explain', 'what is', 'define', 'describe']):
            system_parts.append("Focus on providing clear, comprehensive explanations with examples when helpful.")
        elif any(word in query_lower for word in ['how to', 'steps', 'guide', 'tutorial']):
            system_parts.append("Provide step-by-step guidance and practical instructions.")
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            system_parts.append("Focus on clear comparisons, highlighting key differences and similarities.")
        elif any(word in query_lower for word in ['summarize', 'summary', 'overview']):
            system_parts.append("Provide concise, well-structured summaries hitting the main points.")

        # Add context instructions
        if retrieved_chunks:
            system_parts.append("\nUse the following context from the knowledge base to inform your response:")

            # Format context with source attribution
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                metadata = chunk.get('metadata', {})

                # Build source attribution
                source_info = f"[Source {i}"
                if metadata.get('document_title'):
                    source_info += f": {metadata['document_title']}"
                if metadata.get('page_number'):
                    source_info += f", Page {metadata['page_number']}"
                source_info += f", Relevance: {chunk.get('adjusted_score', chunk.get('score', 0)):.2f}]"

                context_parts.append(f"{source_info}\n{chunk['content']}")

            system_parts.append("\n\n" + "\n\n".join(context_parts))

        # Add response guidelines
        guidelines = [
            "\nResponse Guidelines:",
            "- Be accurate and cite sources when referencing specific information",
            "- If information is insufficient, clearly state what's missing",
            "- Maintain conversation flow and reference previous exchanges when relevant",
            "- Be concise but thorough"
        ]

        if user_preferences.get('detailed_responses'):
            guidelines.append("- Provide detailed explanations with examples")
        if user_preferences.get('prefer_bullet_points'):
            guidelines.append("- Use bullet points and structured formatting when appropriate")

        system_parts.extend(guidelines)

        # Build message list
        messages = [{"role": "system", "content": "\n".join(system_parts)}]

        # Add conversation history (optimized for token usage)
        if conversation_history:
            # Summarize older messages if history is long
            if len(conversation_history) > 6:
                recent_messages = conversation_history[-4:]
                older_messages = conversation_history[:-4]

                # Create summary of older context
                older_summary = "Previous conversation context: " + " ".join([
                    f"{msg['role']}: {msg['content'][:100]}..."
                    for msg in older_messages[-4:]  # Last 4 of the older messages
                ])
                messages.append({"role": "system", "content": older_summary})
                messages.extend(recent_messages)
            else:
                messages.extend(conversation_history)

        # Add current query
        messages.append({"role": "user", "content": user_query})

        return messages

    async def generate_streaming_response(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: int = 1000
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response from OpenAI"""

        try:
            stream = await self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True}
            )

            full_content = ""

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_content += content

                    yield {
                        "type": "content",
                        "content": content,
                        "full_content": full_content
                    }

                # Handle usage info (comes at the end)
                if hasattr(chunk, 'usage') and chunk.usage:
                    yield {
                        "type": "usage",
                        "usage": {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens
                        }
                    }

        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield {
                "type": "error",
                "error": str(e)
            }

    async def process_chat_with_advanced_features(
            self,
            user_id: str,
            message: str,
            session_id: Optional[UUID] = None,
            max_chunks: int = 5,
            temperature: float = 0.7,
            max_tokens: int = 1000,
            user_preferences: Dict[str, Any] = None,
            include_metadata: bool = True,
            stream_response: bool = False
    ) -> Dict[str, Any]:
        """Advanced chat processing with enhanced features"""

        start_time = time.time()
        processing_steps = []

        async with in_transaction():
            try:
                # Step 1: Session management
                step_start = time.time()
                session = await self.create_or_get_session(user_id, session_id)
                processing_steps.append({
                    "step": "session_management",
                    "duration_ms": int((time.time() - step_start) * 1000)
                })

                # Step 2: Get embedding
                step_start = time.time()
                embedding = await self.get_cached_embedding(message)
                processing_steps.append({
                    "step": "embedding_generation",
                    "duration_ms": int((time.time() - step_start) * 1000)
                })

                # Step 3: Conversation context
                step_start = time.time()
                conversation_history = await self.get_conversation_context(session)
                processing_steps.append({
                    "step": "context_retrieval",
                    "duration_ms": int((time.time() - step_start) * 1000)
                })

                # Step 4: Memory management
                step_start = time.time()
                used_chunk_ids = await self.get_session_chunk_memory(session)
                processing_steps.append({
                    "step": "memory_management",
                    "duration_ms": int((time.time() - step_start) * 1000)
                })

                # Step 5: Hybrid retrieval
                step_start = time.time()
                raw_chunks, retrieval_metadata = await self.hybrid_retrieve(
                    query=message,
                    embedding=embedding,
                    exclude_chunk_ids=used_chunk_ids,
                    max_chunks=max_chunks * 2  # Get more for intelligent selection
                )
                processing_steps.append({
                    "step": "hybrid_retrieval",
                    "duration_ms": int((time.time() - step_start) * 1000)
                })

                # Step 6: Intelligent chunk selection
                step_start = time.time()
                selected_chunks = await self.intelligent_chunk_selection(
                    query=message,
                    raw_chunks=raw_chunks,
                    conversation_history=conversation_history,
                    max_chunks=max_chunks
                )
                processing_steps.append({
                    "step": "chunk_selection",
                    "duration_ms": int((time.time() - step_start) * 1000)
                })

                # Step 7: Dynamic prompt building
                step_start = time.time()
                prompt_messages = await self.build_dynamic_prompt(
                    user_query=message,
                    retrieved_chunks=selected_chunks,
                    conversation_history=conversation_history,
                    user_preferences=user_preferences
                )
                processing_steps.append({
                    "step": "prompt_building",
                    "duration_ms": int((time.time() - step_start) * 1000)
                })

                # Step 8: Token counting and optimization
                step_start = time.time()
                prompt_tokens = token_counter.count_message_tokens(prompt_messages)
                available_tokens = token_counter.estimate_response_tokens(max_tokens, prompt_tokens)

                if available_tokens < 100:  # Too little space for response
                    # Trim conversation history
                    if len(conversation_history) > 2:
                        conversation_history = conversation_history[-2:]
                        prompt_messages = await self.build_dynamic_prompt(
                            user_query=message,
                            retrieved_chunks=selected_chunks,
                            conversation_history=conversation_history,
                            user_preferences=user_preferences
                        )
                        prompt_tokens = token_counter.count_message_tokens(prompt_messages)
                        available_tokens = token_counter.estimate_response_tokens(max_tokens, prompt_tokens)

                processing_steps.append({
                    "step": "token_optimization",
                    "duration_ms": int((time.time() - step_start) * 1000),
                    "prompt_tokens": prompt_tokens,
                    "available_tokens": available_tokens
                })

                # Step 9: Response generation
                step_start = time.time()
                if stream_response:
                    # For streaming, return the generator and metadata separately
                    response_generator = self.generate_streaming_response(
                        messages=prompt_messages,
                        temperature=temperature,
                        max_tokens=available_tokens
                    )

                    # Save user message immediately for streaming
                    user_message = await ChatMessage.create(
                        session=session,
                        role="user",
                        content=message,
                        used_chunk_ids=[],
                        retrieval_metadata=retrieval_metadata
                    )

                    return {
                        "stream_generator": response_generator,
                        "session": session,
                        "selected_chunks": selected_chunks,
                        "user_message": user_message,
                        "processing_steps": processing_steps
                    }
                else:
                    # Standard non-streaming response
                    response_text, token_usage = await self.generate_response(
                        messages=prompt_messages,
                        temperature=temperature,
                        max_tokens=available_tokens
                    )

                processing_steps.append({
                    "step": "response_generation",
                    "duration_ms": int((time.time() - step_start) * 1000)
                })

                # Step 10: Database operations
                step_start = time.time()

                # Save user message
                user_message = await ChatMessage.create(
                    session=session,
                    role="user",
                    content=message,
                    used_chunk_ids=[],
                    retrieval_metadata=retrieval_metadata
                )

                # Save assistant response
                chunk_ids = [chunk['chunk_id'] for chunk in selected_chunks]
                chunk_scores = [chunk.get('adjusted_score', chunk['score']) for chunk in selected_chunks]

                assistant_message = await ChatMessage.create(
                    session=session,
                    role="assistant",
                    content=response_text,
                    used_chunk_ids=chunk_ids,
                    retrieval_metadata={
                        **retrieval_metadata,
                        "token_usage": token_usage,
                        "processing_steps": processing_steps
                    },
                    token_count=token_usage.get("total_tokens", 0)
                )

                # Update chunk memory
                if chunk_ids:
                    await self.update_chunk_memory(session, chunk_ids, chunk_scores)

                # Update session
                session.updated_at = datetime.now()
                await session.save()

                processing_steps.append({
                    "step": "database_operations",
                    "duration_ms": int((time.time() - step_start) * 1000)
                })

                # Final response preparation
                total_processing_time = int((time.time() - start_time) * 1000)

                retrieved_chunks_formatted = response_formatter.format_retrieval_results(selected_chunks)

                message_count = await ChatMessage.filter(session=session).count()

                return {
                    "message_id": assistant_message.id,
                    "session_id": session.id,
                    "response": response_text,
                    "retrieved_chunks": retrieved_chunks_formatted if include_metadata else [],
                    "total_chunks_available": len(raw_chunks),
                    "chunks_selected": len(selected_chunks),
                    "token_usage": token_usage,
                    "processing_time_ms": total_processing_time,
                    "processing_steps": processing_steps,
                    "message_count": message_count,
                    "created_at": assistant_message.created_at,
                    "performance_metrics": {
                        "cache_hits": len([step for step in processing_steps if step.get("cache_hit")]),
                        "total_steps": len(processing_steps),
                        "retrieval_efficiency": len(selected_chunks) / max(len(raw_chunks), 1),
                        "token_efficiency": token_usage.get("total_tokens", 0) / max_tokens
                    }
                }

            except Exception as e:
                logger.error(f"Error in enhanced chat processing: {e}", exc_info=True)
                raise

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
                "created_via": "enhanced_api",
                "features": ["intelligent_selection", "dynamic_prompts", "token_optimization"]
            }
        )

        logger.info(f"Created new enhanced chat session {session.id} for user {user_id}")
        return session

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

        # Get current memory with usage limits
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
            max_chunks: int = 10
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Enhanced hybrid retrieval with performance tracking"""

        start_time = time.time()

        try:
            results = await es_client.hybrid_search(
                index_name=ELASTICSEARCH_INDEX_NAME,
                query=query,
                embedding=embedding,
                limit=max_chunks,
                exclude_chunk_ids=exclude_chunk_ids or []
            )

            retrieval_metadata = {
                "query": query,
                "total_found": len(results),
                "excluded_chunks": len(exclude_chunk_ids or []),
                "retrieval_time_ms": int((time.time() - start_time) * 1000),
                "search_type": "enhanced_hybrid",
                "average_score": sum(r['score'] for r in results) / len(results) if results else 0,
                "score_distribution": {
                    "min": min((r['score'] for r in results), default=0),
                    "max": max((r['score'] for r in results), default=0),
                    "std": self._calculate_std([r['score'] for r in results]) if results else 0
                }
            }

            return results, retrieval_metadata

        except Exception as e:
            logger.error(f"Error in enhanced hybrid retrieval: {e}")
            return [], {"error": str(e), "retrieval_time_ms": 0}

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    async def update_chunk_memory(
            self,
            session: ChatSession,
            chunk_ids: List[str],
            scores: List[float]
    ):
        """Enhanced chunk memory management"""

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
                # Update with weighted average of scores
                old_weight = memory.usage_count / (memory.usage_count + 1)
                new_weight = 1 / (memory.usage_count + 1)
                memory.relevance_score = (memory.relevance_score * old_weight + score * new_weight)
                memory.usage_count += 1
                memory.last_used_at = datetime.now()
                await memory.save()

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

    async def analyze_conversation_quality(self, session_id: UUID) -> Dict[str, Any]:
        """Analyze conversation quality metrics"""

        session = await ChatSession.get_or_none(id=session_id)
        if not session:
            return {"error": "Session not found"}

        messages = await ChatMessage.filter(session=session).order_by('created_at')

        if len(messages) < 2:
            return {"error": "Insufficient conversation data"}

        # Calculate metrics
        total_messages = len(messages)
        user_messages = [m for m in messages if m.role == 'user']
        assistant_messages = [m for m in messages if m.role == 'assistant']

        # Token efficiency
        total_tokens = sum(m.token_count or 0 for m in assistant_messages)
        avg_tokens_per_response = total_tokens / len(assistant_messages) if assistant_messages else 0

        # Chunk usage analysis
        all_used_chunks = []
        for msg in assistant_messages:
            all_used_chunks.extend(msg.used_chunk_ids)

        unique_chunks = len(set(all_used_chunks))
        total_chunk_usage = len(all_used_chunks)
        chunk_reuse_rate = (total_chunk_usage - unique_chunks) / max(total_chunk_usage, 1)

        # Response time analysis
        response_times = []
        for msg in assistant_messages:
            if msg.retrieval_metadata and 'processing_steps' in msg.retrieval_metadata:
                total_time = sum(step.get('duration_ms', 0) for step in msg.retrieval_metadata['processing_steps'])
                response_times.append(total_time)

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        return {
            "session_id": str(session_id),
            "conversation_length": total_messages,
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "token_efficiency": {
                "total_tokens": total_tokens,
                "avg_tokens_per_response": round(avg_tokens_per_response, 2),
                "token_efficiency_score": min(1000 / max(avg_tokens_per_response, 1), 1.0)
            },
            "chunk_usage": {
                "unique_chunks_used": unique_chunks,
                "total_chunk_references": total_chunk_usage,
                "chunk_reuse_rate": round(chunk_reuse_rate, 3),
                "avg_chunks_per_response": round(total_chunk_usage / len(assistant_messages),
                                                 2) if assistant_messages else 0
            },
            "performance": {
                "avg_response_time_ms": round(avg_response_time, 2),
                "response_time_score": max(0, 1 - (avg_response_time / 5000))  # 5s baseline
            },
            "overall_quality_score": self._calculate_quality_score(
                avg_tokens_per_response, chunk_reuse_rate, avg_response_time
            )
        }

    def _calculate_quality_score(self, avg_tokens: float, reuse_rate: float, response_time: float) -> float:
        """Calculate overall conversation quality score (0-1)"""

        # Token efficiency score (prefer 200-800 tokens)
        if 200 <= avg_tokens <= 800:
            token_score = 1.0
        elif avg_tokens < 200:
            token_score = avg_tokens / 200
        else:
            token_score = max(0, 1 - (avg_tokens - 800) / 1000)

        # Chunk reuse score (moderate reuse is good)
        if 0.1 <= reuse_rate <= 0.4:
            reuse_score = 1.0
        elif reuse_rate < 0.1:
            reuse_score = reuse_rate / 0.1
        else:
            reuse_score = max(0, 1 - (reuse_rate - 0.4) / 0.6)

        # Response time score (prefer under 2 seconds)
        time_score = max(0, 1 - response_time / 3000)

        # Weighted average
        return round((token_score * 0.4 + reuse_score * 0.3 + time_score * 0.3), 3)


# Global enhanced service instance
enhanced_chat_service = EnhancedChatService()
