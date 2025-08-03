
# conversation/routers.py
from fastapi import APIRouter
from utils.response_wrapper import response_wrapper
from .views import chat_endpoint
from .schemas import ChatResponse

router = APIRouter()

router.post("/conversation/chat", response_model=ChatResponse)(chat_endpoint)
