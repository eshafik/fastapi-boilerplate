import logging
from typing import Tuple
from .schemas import ChatRequest, ChatResponse, UsedChunk
from .redis_manager import RedisSessionManager
from .rag_engine import RAGChatEngine

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, redis_manager: RedisSessionManager, rag_engine: RAGChatEngine):
        self.redis_manager = redis_manager
        self.rag_engine = rag_engine

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Process chat request and return response"""
        try:
            # Get or create session
            session_id = await self.redis_manager.get_or_create_session(request.session_id)

            # Get conversation history and used chunks
            conversation_history = await self.redis_manager.get_conversation_history(session_id)
            used_chunk_ids = await self.redis_manager.get_used_chunk_ids(session_id)

            # Perform hybrid search
            search_results, rag_time_ms = await self.rag_engine.hybrid_search(
                query=request.query,
                top_k=request.top_k,
                exclude_chunks=list(used_chunk_ids),
                bm25_weight=request.bm25_weight,
                vector_weight=request.vector_weight
            )

            # Build prompt with context
            prompt = self.rag_engine.build_prompt(
                user_query=request.query,
                context_chunks=search_results,
                conversation_history=conversation_history
            )

            # Generate response
            response_text, llm_time_ms, input_tokens, output_tokens = await self.rag_engine.generate_response(prompt)

            # Extract chunk IDs for storage
            chunk_ids = [chunk['chunk_id'] for chunk in search_results]

            # Store interaction in Redis
            await self.redis_manager.add_interaction(
                session_id=session_id,
                user_query=request.query,
                assistant_response=response_text,
                chunk_ids=chunk_ids
            )

            # Prepare used chunks for response
            used_chunks = [
                UsedChunk(
                    chunk_id=chunk['chunk_id'],
                    score=chunk['score'],
                    metadata=chunk['metadata']
                )
                for chunk in search_results
            ]

            return ChatResponse(
                response=response_text,
                session_id=session_id,
                rag_time_ms=rag_time_ms,
                llm_time_ms=llm_time_ms,
                tokens_input=input_tokens,
                tokens_output=output_tokens,
                used_chunks=used_chunks
            )

        except Exception as e:
            logger.error(f"Error processing chat request: {e}")
            # Return error response with minimal data
            session_id = request.session_id or "error"
            return ChatResponse(
                response="I apologize, but I encountered an error processing your request. Please try again.",
                session_id=session_id,
                rag_time_ms=0,
                llm_time_ms=0,
                tokens_input=0,
                tokens_output=0,
                used_chunks=[]
            )
