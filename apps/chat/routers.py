
# chat/routers.py
from fastapi import APIRouter
from utils.response_wrapper import response_wrapper
from .views import (send_chat_message, send_chat_message_stream,
                    get_user_sessions, get_chat_history,
                    update_session, delete_session,
                    get_session_chunks, clear_session_memory,
                    get_metrics)

router = APIRouter()

router.post("/chat/message")(response_wrapper(send_chat_message))
router.post("/chat/message/stream")(response_wrapper(send_chat_message_stream))
router.get("/chat/sessions")(response_wrapper(get_user_sessions))
router.get("/chat/sessions/{session_id}/history")(response_wrapper(get_chat_history))
router.put("/chat/sessions/{session_id}")(response_wrapper(update_session))
router.delete("/chat/sessions/{session_id}")(response_wrapper(delete_session))
router.get("/chat/sessions/{session_id}/chunks")(response_wrapper(get_session_chunks))
router.post("/chat/sessions/{session_id}/clear-memory")(response_wrapper(clear_session_memory))
router.get("/chat/metrics")(response_wrapper(get_metrics))
