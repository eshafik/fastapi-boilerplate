import asyncio
import json
import logging
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, Query
from starlette.responses import StreamingResponse

from apps.chat.chat_service import chat_service
from apps.chat.models import ChatSession, ChatMessage, SessionChunkMemory
from apps.chat.schemas import ChatMessageRequestV1, ChatMessageResponseV1, ChatSessionInfo, ChatHistoryResponse

logger = logging.getLogger(__name__)


async def send_chat_message(request: ChatMessageRequestV1):
    """Send a chat message and get AI response with RAG"""

    try:
        result = await chat_service.process_chat_message(
            user_id=request.user_id,
            message=request.message,
            session_id=request.session_id,
            max_chunks=request.max_chunks,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            include_metadata=request.include_metadata
        )

        return ChatMessageResponseV1(**result)

    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")


async def send_chat_message_stream(request: ChatMessageRequestV1):
    """Stream chat response for real-time experience"""

    async def generate_stream():
        try:
            # This is a simplified streaming implementation
            # For full streaming, you'd need to modify the service to support streaming
            result = await chat_service.process_chat_message(
                user_id=request.user_id,
                message=request.message,
                session_id=request.session_id,
                max_chunks=request.max_chunks,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                include_metadata=request.include_metadata
            )

            # Simulate streaming by chunking the response
            response_text = result["response"]
            chunk_size = 20  # words per chunk
            words = response_text.split()

            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if i + chunk_size < len(words):
                    chunk += " "

                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                await asyncio.sleep(0.05)  # Small delay for streaming effect

            # Send final metadata
            final_data = {
                "type": "complete",
                "message_id": str(result["message_id"]),
                "session_id": str(result["session_id"]),
                "token_usage": result["token_usage"],
                "retrieved_chunks": len(result["retrieved_chunks"]),
                "processing_time_ms": result["processing_time_ms"]
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        except Exception as e:
            error_data = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


async def get_user_sessions(
        user_id: str,
        active_only: bool = Query(True, description="Return only active sessions"),
        limit: int = Query(50, ge=1, le=100)
):
    """Get chat sessions for a user"""

    try:
        filters = {"user_id": user_id}
        if active_only:
            filters["is_active"] = True

        sessions = await ChatSession.filter(**filters).order_by('-updated_at').limit(limit)

        result = []
        for session in sessions:
            message_count = await ChatMessage.filter(session=session).count()
            result.append(ChatSessionInfo(
                session_id=session.id,
                user_id=session.user_id,
                session_name=session.session_name,
                message_count=message_count,
                created_at=session.created_at,
                updated_at=session.updated_at,
                is_active=session.is_active
            ))

        return result

    except Exception as e:
        logger.error(f"Error getting user sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")


async def get_chat_history(
        session_id: UUID,
        user_id: str,
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100)
):
    """Get chat history for a session with pagination"""

    try:
        # Verify session ownership
        session = await ChatSession.get_or_none(id=session_id, user_id=user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get total message count
        total_messages = await ChatMessage.filter(session=session).count()

        # Get paginated messages
        offset = (page - 1) * page_size
        messages = await ChatMessage.filter(
            session=session
        ).order_by('created_at').offset(offset).limit(page_size)

        # Format messages
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "id": str(msg.id),
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
                "used_chunks": len(msg.used_chunk_ids),
                "token_count": msg.token_count
            })

        # Get session info
        message_count = await ChatMessage.filter(session=session).count()
        session_info = ChatSessionInfo(
            session_id=session.id,
            user_id=session.user_id,
            session_name=session.session_name,
            message_count=message_count,
            created_at=session.created_at,
            updated_at=session.updated_at,
            is_active=session.is_active
        )

        return ChatHistoryResponse(
            session=session_info,
            messages=formatted_messages,
            total_messages=total_messages,
            page=page,
            page_size=page_size
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")


async def update_session(
        session_id: UUID,
        user_id: str,
        session_name: Optional[str] = None,
        is_active: Optional[bool] = None
):
    """Update session properties"""

    try:
        session = await ChatSession.get_or_none(id=session_id, user_id=user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if session_name is not None:
            session.session_name = session_name

        if is_active is not None:
            session.is_active = is_active

        await session.save()

        return {"message": "Session updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session: {e}")
        raise HTTPException(status_code=500, detail="Failed to update session")


async def delete_session(session_id: UUID, user_id: str):
    """Delete a chat session and all its messages"""

    try:
        session = await ChatSession.get_or_none(id=session_id, user_id=user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Delete related data
        await ChatMessage.filter(session=session).delete()
        await SessionChunkMemory.filter(session=session).delete()
        await session.delete()

        return {"message": "Session deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")


async def get_session_chunks(
        session_id: UUID,
        user_id: str,
        include_content: bool = Query(False, description="Include chunk content in response")
):
    """Get chunks used in a session with usage statistics"""

    try:
        session = await ChatSession.get_or_none(id=session_id, user_id=user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        chunks = await SessionChunkMemory.filter(session=session).order_by('-usage_count')

        result = []
        for chunk in chunks:
            chunk_info = {
                "chunk_id": chunk.chunk_id,
                "usage_count": chunk.usage_count,
                "relevance_score": chunk.relevance_score,
                "first_used_at": chunk.first_used_at.isoformat(),
                "last_used_at": chunk.last_used_at.isoformat()
            }

            # Optionally include content
            if include_content:
                try:
                    # You could fetch content from Elasticsearch here
                    # For now, just indicate it's available
                    chunk_info["content_available"] = True
                except:
                    chunk_info["content_available"] = False

            result.append(chunk_info)

        return {
            "session_id": str(session_id),
            "total_unique_chunks": len(result),
            "chunks": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session chunks: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session chunks")


async def clear_session_memory(session_id: UUID, user_id: str):
    """Clear chunk memory for a session to allow fresh retrievals"""

    try:
        session = await ChatSession.get_or_none(id=session_id, user_id=user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        deleted_count = await SessionChunkMemory.filter(session=session).delete()

        return {
            "message": f"Cleared memory for {deleted_count} chunks",
            "session_id": str(session_id)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing session memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear session memory")


async def get_metrics():
    """Basic metrics endpoint"""

    try:
        # Get basic statistics
        total_sessions = await ChatSession.all().count()
        active_sessions = await ChatSession.filter(is_active=True).count()
        total_messages = await ChatMessage.all().count()
        total_chunks_in_memory = await SessionChunkMemory.all().count()

        # Get recent activity (last 24 hours)
        from datetime import datetime, timedelta
        yesterday = datetime.now() - timedelta(days=1)

        recent_sessions = await ChatSession.filter(created_at__gte=yesterday).count()
        recent_messages = await ChatMessage.filter(created_at__gte=yesterday).count()

        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "total_messages": total_messages,
            "chunks_in_memory": total_chunks_in_memory,
            "recent_activity": {
                "sessions_24h": recent_sessions,
                "messages_24h": recent_messages
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics")