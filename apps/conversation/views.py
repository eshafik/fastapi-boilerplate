from fastapi import APIRouter, HTTPException, Depends
import logging

from config.settings import OPENAI_API_KEY, ELASTICSEARCH_DEFAULT_INDEX_NAME, REDIS_URL
from .schemas import ChatRequest, ChatResponse
from .service import ChatService
from .redis_manager import RedisSessionManager
from .rag_engine import RAGChatEngine


logger = logging.getLogger(__name__)


async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint for conversational RAG system.

    Features:
    - Hybrid search (BM25 + vector similarity)
    - Session-based conversation memory
    - Performance metrics tracking
    - Cost-optimized prompting
    """
    redis_manager = RedisSessionManager(redis_url=REDIS_URL)
    rag_engine = RAGChatEngine(openai_api_key=OPENAI_API_KEY, index_name=ELASTICSEARCH_DEFAULT_INDEX_NAME)
    chat_service = ChatService(redis_manager=redis_manager, rag_engine=rag_engine)

    logger.info(f"Processing chat request for session: {request.session_id}")

    try:
        response = await chat_service.process_chat(request)

        logger.info(
            f"Chat processed successfully. Session: {response.session_id}, "
            f"RAG: {response.rag_time_ms}ms, LLM: {response.llm_time_ms}ms, "
            f"Tokens: {response.tokens_input}/{response.tokens_output}"
        )
        await redis_manager.close()

        return response

    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        await redis_manager.close()
        raise HTTPException(status_code=500, detail="Internal server error")