# chat/enhanced_chat_service.py
import asyncio
import time
import json
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from uuid import UUID, uuid4
import logging
from datetime import datetime, timedelta

from openai import AsyncOpenAI
from tortoise.transactions import in_transaction

from apps.chat.models import ChatSession, ChatMessage, SessionChunkMemory
from apps.document.elasticsearch_client import es_client
from config.settings import OPENAI_API_KEY, ELASTICSEARCH_DEFAULT_INDEX_NAME
from utils.response_formatter import response_formatter

logger = logging.getLogger(__name__)

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
from utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class EnhancedChatService:
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-4o-mini"

        # Initialize cache manager
        self.cache_manager = CacheManager()

        # Advanced configuration
        self.max_conversation_history = 10
        self.chunk_memory_decay_days = 7
        self.max_chunk_reuse_count = 3

        # Quality settings
        self.min_chunk_score_threshold = 0.1
        self.semantic_similarity_weight = 0.7
        self.keyword_match_weight = 0.3

        # Performance settings
        self.max_concurrent_requests = 10
        self.request_timeout = 30

    async def get_cached_embedding(self, text: str) -> List[float]:
        """Get embedding with optimized caching"""

        cache_key = f"embedding:{hash(text)}"

        # Try cache first
        cached_embedding = await self.cache_manager.get(cache_key)
        if cached_embedding:
            return cached_embedding

        # Get new embedding
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding

            # Cache with 1 hour TTL
            await self.cache_manager.set(cache_key, embedding, ttl=3600)

            return embedding

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    async def intelligent_chunk_selection(
            self,
            query: str,
            raw_chunks: List[Dict[str, Any]],
            conversation_history: List[Dict[str, str]],
            session_memory: Dict[str, float],
            max_chunks: int = 5
    ) -> List[Dict[str, Any]]:
        """Intelligently select the most relevant chunks with advanced scoring"""

        if not raw_chunks:
            return []

        scored_chunks = []

        for chunk in raw_chunks:
            base_score = chunk['score']
            chunk_id = chunk['chunk_id']

            # Apply memory boost if chunk was previously useful
            memory_boost = session_memory.get(chunk_id, 0) * 0.1

            # Recency boost for newer documents
            metadata = chunk.get('metadata', {})
            recency_boost = 0
            if 'created_at' in metadata:
                try:
                    doc_date = datetime.fromisoformat(metadata['created_at'])
                    doc_age_days = (datetime.now() - doc_date).days
                    recency_boost = max(0, 1 - (doc_age_days / 365)) * 0.05
                except:
                    pass

            # Context relevance boost
            context_boost = 0
            if conversation_history:
                recent_context = " ".join([msg['content'] for msg in conversation_history[-3:]])
                chunk_words = set(chunk['content'].lower().split())
                context_words = set(recent_context.lower().split())
                overlap = len(chunk_words.intersection(context_words))
                if overlap > 0:
                    context_boost = min(overlap * 0.03, 0.15)

            # Diversity penalty for very similar chunks
            diversity_penalty = 0
            for existing_chunk in scored_chunks:
                existing_words = set(existing_chunk['content'].lower().split())
                current_words = set(chunk['content'].lower().split())
                similarity = len(existing_words.intersection(current_words)) / len(existing_words.union(current_words))
                if similarity > 0.8:
                    diversity_penalty = min(diversity_penalty + 0.1, 0.3)

            # Calculate final score
            final_score = base_score + memory_boost + recency_boost + context_boost - diversity_penalty

            # Apply minimum threshold
            if final_score >= self.min_chunk_score_threshold:
                scored_chunks.append({
                    **chunk,
                    'adjusted_score': final_score,
                    'score_breakdown': {
                        'base_score': base_score,
                        'memory_boost': memory_boost,
                        'recency_boost': recency_boost,
                        'context_boost': context_boost,
                        'diversity_penalty': diversity_penalty
                    }
                })

        # Sort by adjusted score and take top chunks
        scored_chunks.sort(key=lambda x: x['adjusted_score'], reverse=True)
        return scored_chunks[:max_chunks]

    async def build_optimized_prompt(
            self,
            user_query: str,
            retrieved_chunks: List[Dict[str, Any]],
            conversation_history: List[Dict[str, str]],
            user_preferences: Dict[str, Any] = None,
            max_context_tokens: int = 3000
    ) -> List[Dict[str, str]]:
        """Build optimized prompt with token management"""

        user_preferences = user_preferences or {}

        # Adaptive system message
        system_parts = [
            "You are an expert AI assistant with access to a comprehensive knowledge base.",
            "Provide accurate, helpful, and contextually relevant responses based on the retrieved information."
        ]

        # Query type analysis for specialized instructions
        query_lower = user_query.lower()
        query_indicators = {
            'explanation': ['explain', 'what is', 'define', 'describe', 'clarify'],
            'instruction': ['how to', 'steps', 'guide', 'tutorial', 'process'],
            'comparison': ['compare', 'difference', 'versus', 'vs', 'better'],
            'summary': ['summarize', 'summary', 'overview', 'brief'],
            'analysis': ['analyze', 'evaluate', 'assess', 'examine']
        }

        query_type = 'general'
        for qtype, indicators in query_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                query_type = qtype
                break

        # Add type-specific instructions
        type_instructions = {
            'explanation': "Focus on clear, comprehensive explanations with examples when helpful.",
            'instruction': "Provide step-by-step guidance with actionable instructions.",
            'comparison': "Highlight key differences and similarities in a structured format.",
            'summary': "Provide concise, well-organized summaries covering main points.",
            'analysis': "Offer thorough analysis with reasoning and evidence."
        }

        if query_type in type_instructions:
            system_parts.append(type_instructions[query_type])

        # Add context from retrieved chunks
        if retrieved_chunks:
            system_parts.append("\n**Knowledge Base Context:**")

            context_parts = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                metadata = chunk.get('metadata', {})

                # Build concise source attribution
                source_info = f"[{i}]"
                if metadata.get('document_title'):
                    source_info += f" {metadata['document_title']}"
                if metadata.get('page_number'):
                    source_info += f" (p.{metadata['page_number']})"

                # Truncate very long chunks to manage tokens
                content = chunk['content']
                if len(content) > 800:
                    content = content[:750] + "..."

                context_parts.append(f"{source_info}: {content}")

            system_parts.append("\n".join(context_parts))

        # Add response guidelines
        guidelines = [
            "\n**Response Guidelines:**",
            "• Cite sources using [1], [2], etc. when referencing specific information",
            "• If information is insufficient, clearly state limitations",
            "• Maintain conversational flow and build on previous exchanges",
            "• Be precise but comprehensive"
        ]

        # User preference adjustments
        if user_preferences.get('detailed_responses'):
            guidelines.append("• Provide detailed explanations with examples and context")
        if user_preferences.get('structured_format'):
            guidelines.append("• Use clear structure with headings and bullet points")
        if user_preferences.get('technical_depth'):
            guidelines.append("• Include technical details and implementation specifics")

        system_parts.extend(guidelines)

        # Build initial message list
        messages = [{"role": "system", "content": "\n".join(system_parts)}]

        # Add optimized conversation history
        if conversation_history:
            # Estimate tokens and optimize history length
            system_tokens = token_counter.count_message_tokens([messages[0]])
            available_tokens = max_context_tokens - system_tokens - 200  # Reserve for user query

            # Add conversation history within token limits
            history_messages = []
            current_tokens = 0

            # Start from most recent and work backwards
            for msg in reversed(conversation_history):
                msg_tokens = token_counter.count_message_tokens([msg])
                if current_tokens + msg_tokens > available_tokens:
                    break
                history_messages.insert(0, msg)
                current_tokens += msg_tokens

            messages.extend(history_messages)

        # Add current user query
        messages.append({"role": "user", "content": user_query})

        return messages

    async def generate_streaming_response(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: int = 1000
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response with error handling"""

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
            chunk_count = 0

            async for chunk in stream:
                chunk_count += 1

                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_content += content

                    yield {
                        "type": "content",
                        "content": content,
                        "full_content": full_content,
                        "chunk_index": chunk_count
                    }

                # Handle usage info
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
                "error": str(e),
                "error_type": type(e).__name__
            }

    async def process_chat_with_advanced_features(
            self,
            user_id: str,
            message: str,
            session_id: Optional[UUID] = None,
            session_name: Optional[str] = None,
            max_chunks: int = 5,
            temperature: float = 0.7,
            max_tokens: int = 1000,
            user_preferences: Dict[str, Any] = None,
            include_metadata: bool = True,
            stream_response: bool = False
    ) -> Dict[str, Any]:
        """Main chat processing with comprehensive features"""

        start_time = time.time()
        processing_steps = []
        user_preferences = user_preferences or {}

        async with in_transaction():
            try:
                # Step 1: Session management
                step_start = time.time()
                session = await self.create_or_get_session(
                    user_id=user_id,
                    session_id=session_id,
                    session_name=session_name
                )
                processing_steps.append({
                    "step": "session_management",
                    "duration_ms": int((time.time() - step_start) * 1000)
                })

                # Step 2: Get embedding with caching
                step_start = time.time()
                embedding = await self.get_cached_embedding(message)
                cache_hit = await self.cache_manager.exists(f"embedding:{hash(message)}")
                processing_steps.append({
                    "step": "embedding_generation",
                    "duration_ms": int((time.time() - step_start) * 1000),
                    "cache_hit": cache_hit
                })

                # Step 3: Get conversation context
                step_start = time.time()
                conversation_history = await self.get_conversation_context(session)
                processing_steps.append({
                    "step": "context_retrieval",
                    "duration_ms": int((time.time() - step_start) * 1000)
                })

                # Step 4: Session memory management
                step_start = time.time()
                used_chunk_ids = await self.get_session_chunk_memory(session)
                session_memory = await self.get_session_memory_scores(session)
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
                    session_memory=session_memory,
                    max_chunks=max_chunks
                )
                processing_steps.append({
                    "step": "chunk_selection",
                    "duration_ms": int((time.time() - step_start) * 1000)
                })

                # Step 7: Optimized prompt building
                step_start = time.time()
                prompt_messages = await self.build_optimized_prompt(
                    user_query=message,
                    retrieved_chunks=selected_chunks,
                    conversation_history=conversation_history,
                    user_preferences=user_preferences,
                    max_context_tokens=3500
                )
                processing_steps.append({
                    "step": "prompt_building",
                    "duration_ms": int((time.time() - step_start) * 1000)
                })

                # Step 8: Token optimization
                step_start = time.time()
                prompt_tokens = token_counter.count_message_tokens(prompt_messages)
                available_tokens = min(max_tokens, 4000 - prompt_tokens - 100)  # Safety margin

                if available_tokens < 100:
                    # Aggressive optimization if needed
                    conversation_history = conversation_history[-2:] if len(
                        conversation_history) > 2 else conversation_history
                    prompt_messages = await self.build_optimized_prompt(
                        user_query=message,
                        retrieved_chunks=selected_chunks[:3],  # Reduce chunks too
                        conversation_history=conversation_history,
                        user_preferences=user_preferences,
                        max_context_tokens=2500
                    )
                    prompt_tokens = token_counter.count_message_tokens(prompt_messages)
                    available_tokens = min(max_tokens, 4000 - prompt_tokens - 100)

                processing_steps.append({
                    "step": "token_optimization",
                    "duration_ms": int((time.time() - step_start) * 1000),
                    "prompt_tokens": prompt_tokens,
                    "available_tokens": available_tokens
                })

                # Step 9: Response generation
                step_start = time.time()

                if stream_response:
                    # Return streaming generator and metadata
                    response_generator = self.generate_streaming_response(
                        messages=prompt_messages,
                        temperature=temperature,
                        max_tokens=available_tokens
                    )

                    # Save user message for streaming
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
                    # Standard response generation
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
                        "processing_steps": processing_steps,
                        "chunk_selection_details": [
                            {
                                "chunk_id": chunk['chunk_id'],
                                "score_breakdown": chunk.get('score_breakdown', {})
                            } for chunk in selected_chunks
                        ] if include_metadata else []
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

                # Prepare final response
                total_processing_time = int((time.time() - start_time) * 1000)
                message_count = await ChatMessage.filter(session=session).count()

                # Format retrieved chunks for response
                formatted_chunks = []
                if include_metadata:
                    for chunk in selected_chunks:
                        formatted_chunks.append({
                            "chunk_id": chunk['chunk_id'],
                            "content": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk[
                                'content'],
                            "score": chunk.get('adjusted_score', chunk['score']),
                            "metadata": chunk.get('metadata', {})
                        })

                return {
                    "message_id": assistant_message.id,
                    "session_id": session.id,
                    "response": response_text,
                    "retrieved_chunks": formatted_chunks,
                    "total_chunks_available": len(raw_chunks),
                    "chunks_selected": len(selected_chunks),
                    "token_usage": {
                        "prompt_tokens": token_usage.get("prompt_tokens", 0),
                        "completion_tokens": token_usage.get("completion_tokens", 0),
                        "total_tokens": token_usage.get("total_tokens", 0)
                    },
                    "processing_time_ms": total_processing_time,
                    "processing_steps": processing_steps,
                    "message_count": message_count,
                    "created_at": assistant_message.created_at,
                    "performance_metrics": {
                        "cache_hits": len([step for step in processing_steps if step.get("cache_hit")]),
                        "total_steps": len(processing_steps),
                        "retrieval_efficiency": len(selected_chunks) / max(len(raw_chunks), 1),
                        "token_efficiency": token_usage.get("total_tokens", 0) / max(max_tokens, 1)
                    }
                }

            except Exception as e:
                logger.error(f"Error in enhanced chat processing: {e}", exc_info=True)
                raise

    async def create_or_get_session(
            self,
            user_id: str,
            session_id: Optional[UUID] = None,
            session_name: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None
    ) -> ChatSession:
        """Create new session or retrieve existing one with enhanced configuration"""

        if session_id:
            session = await ChatSession.get_or_none(id=session_id, user_id=user_id, is_active=True)
            if session:
                return session

        # Create new session with optimized defaults
        default_config = {
            "max_chunks": 5,
            "temperature": 0.7,
            "max_tokens": 1000,
            "created_via": "rest_api",
            "features": [
                "intelligent_selection",
                "optimized_prompts",
                "token_optimization",
                "memory_management",
                "performance_tracking"
            ],
            "version": "2.0"
        }

        if config:
            default_config.update(config)

        session = await ChatSession.create(
            user_id=user_id,
            session_name=session_name or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            config=default_config
        )

        logger.info(f"Created new enhanced chat session {session.id} for user {user_id}")
        return session

    async def get_conversation_context(self, session: ChatSession, limit: int = None) -> List[Dict[str, str]]:
        """Get optimized conversation history"""
        limit = limit or self.max_conversation_history

        messages = await ChatMessage.filter(
            session=session
        ).order_by('-created_at').limit(limit).values(
            'role', 'content', 'created_at', 'token_count'
        )

        # Return in chronological order with token-aware truncation
        context_messages = []
        total_tokens = 0
        max_context_tokens = 2000

        for msg in reversed(messages):
            # Estimate tokens for this message
            msg_tokens = msg.get('token_count', len(msg['content']) // 4)

            if total_tokens + msg_tokens > max_context_tokens and context_messages:
                break

            context_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
            total_tokens += msg_tokens

        return context_messages

    async def get_session_chunk_memory(self, session: ChatSession) -> List[str]:
        """Get chunk IDs to avoid immediate reuse"""

        # Clean up old memory
        cutoff_date = datetime.now() - timedelta(days=self.chunk_memory_decay_days)
        await SessionChunkMemory.filter(
            session=session,
            last_used_at__lt=cutoff_date
        ).delete()

        # Get chunks that haven't been overused
        memory = await SessionChunkMemory.filter(
            session=session,
            usage_count__lt=self.max_chunk_reuse_count
        ).values_list('chunk_id', flat=True)

        return list(memory)

    async def get_session_memory_scores(self, session: ChatSession) -> Dict[str, float]:
        """Get memory scores for intelligent selection"""

        memory_records = await SessionChunkMemory.filter(session=session).values(
            'chunk_id', 'relevance_score', 'usage_count'
        )

        return {
            record['chunk_id']: record['relevance_score'] * (1 + record['usage_count'] * 0.1)
            for record in memory_records
        }

    async def hybrid_retrieve(
            self,
            query: str,
            embedding: List[float],
            exclude_chunk_ids: Optional[List[str]] = None,
            max_chunks: int = 10
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Enhanced hybrid retrieval with comprehensive metadata"""

        start_time = time.time()

        try:
            results = await es_client.hybrid_search(
                index_name=ELASTICSEARCH_INDEX_NAME,
                query=query,
                embedding=embedding,
                limit=max_chunks,
                exclude_chunk_ids=exclude_chunk_ids or []
            )

            scores = [r['score'] for r in results] if results else []

            retrieval_metadata = {
                "query": query,
                "total_found": len(results),
                "excluded_chunks": len(exclude_chunk_ids or []),
                "retrieval_time_ms": int((time.time() - start_time) * 1000),
                "search_type": "enhanced_hybrid",
                "score_statistics": {
                    "average": sum(scores) / len(scores) if scores else 0,
                    "min": min(scores) if scores else 0,
                    "max": max(scores) if scores else 0,
                    "std": self._calculate_std(scores) if len(scores) > 1 else 0
                },
                "elasticsearch_index": ELASTICSEARCH_INDEX_NAME
            }

            return results, retrieval_metadata

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return [], {
                "error": str(e),
                "retrieval_time_ms": int((time.time() - start_time) * 1000),
                "search_type": "failed"
            }

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
        """Enhanced chunk memory with weighted scoring"""

        for chunk_id, score in zip(chunk_ids, scores):
            memory, created = await SessionChunkMemory.get_or_create(
                session=session,
                chunk_id=chunk_id,
                defaults={
                    "relevance_score": score,
                    "usage_count": 1,
                    "first_used_at": datetime.now(),
                    "last_used_at": datetime.now()
                }
            )

            if not created:
                # Exponential moving average for relevance score
                alpha = 0.3  # Learning rate
                memory.relevance_score = (1 - alpha) * memory.relevance_score + alpha * score
                memory.usage_count += 1
                memory.last_used_at = datetime.now()
                await memory.save()

    async def generate_response(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: int = 1000
    ) -> Tuple[str, Dict[str, int]]:
        """Generate response with timeout and retry logic"""

        for attempt in range(3):  # Retry up to 3 times
            try:
                response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=self.chat_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False
                    ),
                    timeout=self.request_timeout
                )

                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

                return response.choices[0].message.content, token_usage

            except asyncio.TimeoutError:
                logger.warning(f"OpenAI request timeout on attempt {attempt + 1}")
                if attempt == 2:  # Last attempt
                    raise
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
            except Exception as e:
                logger.error(f"Error generating response on attempt {attempt + 1}: {e}")
                if attempt == 2:  # Last attempt
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))

    async def analyze_conversation_quality(self, session_id: UUID) -> Dict[str, Any]:
        """Comprehensive conversation quality analysis"""

        session = await ChatSession.get_or_none(id=session_id)
        if not session:
            return {"error": "Session not found"}

        messages = await ChatMessage.filter(session=session).order_by('created_at')

        if len(messages) < 2:
            return {"error": "Insufficient conversation data"}

        # Separate user and assistant messages
        user_messages = [m for m in messages if m.role == 'user']
        assistant_messages = [m for m in messages if m.role == 'assistant']

        # Token efficiency analysis
        total_tokens = sum(m.token_count or 0 for m in assistant_messages)
        avg_tokens_per_response = total_tokens / len(assistant_messages) if assistant_messages else 0

        # Chunk usage analysis
        all_used_chunks = []
        chunk_reuse_data = {}

        for msg in assistant_messages:
            for chunk_id in msg.used_chunk_ids:
                all_used_chunks.append(chunk_id)
                chunk_reuse_data[chunk_id] = chunk_reuse_data.get(chunk_id, 0) + 1

        unique_chunks = len(set(all_used_chunks))
        total_chunk_usage = len(all_used_chunks)
        chunk_reuse_rate = (total_chunk_usage - unique_chunks) / max(total_chunk_usage, 1)

        # Performance analysis
        response_times = []
        processing_efficiency = []

        for msg in assistant_messages:
            if msg.retrieval_metadata and 'processing_steps' in msg.retrieval_metadata:
                steps = msg.retrieval_metadata['processing_steps']
                total_time = sum(step.get('duration_ms', 0) for step in steps)
                response_times.append(total_time)

                # Calculate processing efficiency
                retrieval_time = sum(step.get('duration_ms', 0) for step in steps
                                     if step.get('step') in ['hybrid_retrieval', 'chunk_selection'])
                generation_time = sum(step.get('duration_ms', 0) for step in steps
                                      if step.get('step') == 'response_generation')

                if total_time > 0:
                    efficiency = (retrieval_time + generation_time) / total_time
                    processing_efficiency.append(efficiency)

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        avg_efficiency = sum(processing_efficiency) / len(processing_efficiency) if processing_efficiency else 0

        # Quality scoring
        quality_scores = {
            "token_efficiency": self._score_token_efficiency(avg_tokens_per_response),
            "chunk_utilization": self._score_chunk_utilization(chunk_reuse_rate),
            "response_speed": self._score_response_speed(avg_response_time),
            "processing_efficiency": self._score_processing_efficiency(avg_efficiency)
        }

        overall_quality_score = sum(quality_scores.values()) / len(quality_scores)

        return {
            "session_id": str(session_id),
            "conversation_length": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "token_efficiency": {
                "total_tokens": total_tokens,
                "avg_tokens_per_response": round(avg_tokens_per_response, 2),
                "token_efficiency_score": quality_scores["token_efficiency"]
            },
            "chunk_usage": {
                "unique_chunks_used": unique_chunks,
                "total_chunk_references": total_chunk_usage,
                "chunk_reuse_rate": round(chunk_reuse_rate, 3),
                "avg_chunks_per_response": round(total_chunk_usage / len(assistant_messages),
                                                 2) if assistant_messages else 0,
                "most_reused_chunks": sorted(chunk_reuse_data.items(), key=lambda x: x[1], reverse=True)[:5]
            },
            "performance": {
                "avg_response_time_ms": round(avg_response_time, 2),
                "response_time_score": quality_scores["response_speed"],
                "processing_efficiency": round(avg_efficiency, 3),
                "efficiency_score": quality_scores["processing_efficiency"]
            },
            "quality_breakdown": quality_scores,
            "overall_quality_score": round(overall_quality_score, 3),
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _score_token_efficiency(self, avg_tokens: float) -> float:
        """Score token efficiency (0-1, higher is better)"""
        if 200 <= avg_tokens <= 800:
            return 1.0
        elif avg_tokens < 200:
            return max(0.5, avg_tokens / 200)
        else:
            return max(0.1, 1 - (avg_tokens - 800) / 1200)

    def _score_chunk_utilization(self, reuse_rate: float) -> float:
        """Score chunk utilization (0-1, moderate reuse is optimal)"""
        if 0.15 <= reuse_rate <= 0.35:
            return 1.0
        elif reuse_rate < 0.15:
            return max(0.3, reuse_rate / 0.15)
        else:
            return max(0.2, 1 - (reuse_rate - 0.35) / 0.65)

    def _score_response_speed(self, avg_time_ms: float) -> float:
        """Score response speed (0-1, faster is better)"""
        if avg_time_ms <= 1000:
            return 1.0
        elif avg_time_ms <= 3000:
            return 1 - (avg_time_ms - 1000) / 2000 * 0.5
        else:
            return max(0.1, 0.5 - (avg_time_ms - 3000) / 5000 * 0.4)

    def _score_processing_efficiency(self, efficiency: float) -> float:
        """Score processing efficiency (0-1, higher is better)"""
        return min(1.0, max(0.1, efficiency))

    async def get_user_chat_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user chat statistics"""

        # Get all user sessions
        sessions = await ChatSession.filter(user_id=user_id)
        if not sessions:
            return {"error": "No sessions found for user"}

        # Aggregate statistics
        total_sessions = len(sessions)
        active_sessions = len([s for s in sessions if s.is_active])

        # Get all messages for the user
        all_messages = []
        for session in sessions:
            session_messages = await ChatMessage.filter(session=session)
            all_messages.extend(session_messages)

        if not all_messages:
            return {"error": "No messages found"}

        # Analyze message patterns
        user_messages = [m for m in all_messages if m.role == 'user']
        assistant_messages = [m for m in all_messages if m.role == 'assistant']

        # Token usage analysis
        total_tokens = sum(m.token_count or 0 for m in assistant_messages)
        avg_tokens_per_message = total_tokens / len(assistant_messages) if assistant_messages else 0

        # Usage patterns
        message_lengths = [len(m.content) for m in user_messages]
        avg_message_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0

        # Time-based analysis
        if all_messages:
            first_message = min(all_messages, key=lambda x: x.created_at)
            last_message = max(all_messages, key=lambda x: x.created_at)
            usage_span_days = (last_message.created_at - first_message.created_at).days + 1
        else:
            usage_span_days = 0

        return {
            "user_id": user_id,
            "session_statistics": {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "avg_messages_per_session": round(len(all_messages) / total_sessions, 2) if total_sessions else 0
            },
            "message_statistics": {
                "total_messages": len(all_messages),
                "user_messages": len(user_messages),
                "assistant_messages": len(assistant_messages),
                "avg_user_message_length": round(avg_message_length, 2),
                "total_tokens_used": total_tokens,
                "avg_tokens_per_response": round(avg_tokens_per_message, 2)
            },
            "usage_patterns": {
                "usage_span_days": usage_span_days,
                "messages_per_day": round(len(all_messages) / max(usage_span_days, 1), 2),
                "most_active_session": str(max(sessions, key=lambda s: len(
                    [m for m in all_messages if m.session_id == s.id])).id) if sessions else None
            },
            "analysis_timestamp": datetime.now().isoformat()
        }

    async def cleanup_inactive_sessions(self, user_id: str, days_inactive: int = 30) -> Dict[str, Any]:
        """Clean up inactive sessions and their data"""

        cutoff_date = datetime.now() - timedelta(days=days_inactive)

        # Find inactive sessions
        inactive_sessions = await ChatSession.filter(
            user_id=user_id,
            updated_at__lt=cutoff_date,
            is_active=True
        )

        if not inactive_sessions:
            return {
                "cleaned_sessions": 0,
                "cleaned_messages": 0,
                "cleaned_memory_records": 0
            }

        session_ids = [s.id for s in inactive_sessions]

        # Count what will be deleted
        message_count = await ChatMessage.filter(session_id__in=session_ids).count()
        memory_count = await SessionChunkMemory.filter(session_id__in=session_ids).count()

        # Perform cleanup
        await ChatMessage.filter(session_id__in=session_ids).delete()
        await SessionChunkMemory.filter(session_id__in=session_ids).delete()

        # Mark sessions as inactive instead of deleting
        for session in inactive_sessions:
            session.is_active = False
            await session.save()

        logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions for user {user_id}")

        return {
            "cleaned_sessions": len(inactive_sessions),
            "cleaned_messages": message_count,
            "cleaned_memory_records": memory_count,
            "cleanup_date": datetime.now().isoformat()
        }

    async def export_conversation_data(self, session_id: UUID, user_id: str, include_metadata: bool = False) -> Dict[
        str, Any]:
        """Export conversation data for backup or analysis"""

        session = await ChatSession.get_or_none(id=session_id, user_id=user_id)
        if not session:
            raise ValueError("Session not found")

        messages = await ChatMessage.filter(session=session).order_by('created_at')

        exported_data = {
            "session_info": {
                "id": str(session.id),
                "user_id": session.user_id,
                "session_name": session.session_name,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "config": session.config
            },
            "messages": [],
            "export_metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_messages": len(messages),
                "include_metadata": include_metadata
            }
        }

        for msg in messages:
            message_data = {
                "id": str(msg.id),
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
                "token_count": msg.token_count
            }

            if include_metadata and msg.role == "assistant":
                message_data["used_chunks"] = msg.used_chunk_ids
                message_data["retrieval_metadata"] = msg.retrieval_metadata

            exported_data["messages"].append(message_data)

        return exported_data


# Utility classes for enhanced functionality
class CacheManager:
    """Simple in-memory cache manager with TTL support"""

    def __init__(self):
        self._cache = {}
        self._expiry = {}

    async def get(self, key: str):
        """Get value from cache if not expired"""
        if key in self._cache:
            if key not in self._expiry or self._expiry[key] > time.time():
                return self._cache[key]
            else:
                # Expired, remove
                del self._cache[key]
                del self._expiry[key]
        return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL"""
        self._cache[key] = value
        self._expiry[key] = time.time() + ttl

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        return await self.get(key) is not None

    async def clear_expired(self):
        """Clear expired entries"""
        current_time = time.time()
        expired_keys = [k for k, expiry in self._expiry.items() if expiry <= current_time]
        for key in expired_keys:
            if key in self._cache:
                del self._cache[key]
            del self._expiry[key]


# Global enhanced service instance
enhanced_chat_service = EnhancedChatService()

