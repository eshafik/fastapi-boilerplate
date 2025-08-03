# middleware.py
import time
from typing import Dict, Tuple, Any
from fastapi import HTTPException, Request
from collections import defaultdict, deque
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Modify the request here (e.g., add custom headers, authentication, etc.)
        request.state.custom_attribute = "Custom value"

        response: Response = await call_next(request)
        # Modify the response here (if needed)
        return response


class RequestValidator:
    """Validate and sanitize requests"""

    @staticmethod
    def validate_message(message: str) -> str:
        """Validate and clean user message"""

        if not message or not message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Clean the message
        message = message.strip()

        # Check length
        if len(message) > 10000:
            raise HTTPException(status_code=400, detail="Message too long (max 10,000 characters)")

        # Basic content filtering (extend as needed)
        forbidden_patterns = [
            # Add patterns for content you want to block
        ]

        message_lower = message.lower()
        for pattern in forbidden_patterns:
            if pattern in message_lower:
                raise HTTPException(status_code=400, detail="Message contains prohibited content")

        return message

    @staticmethod
    def validate_session_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate session configuration"""

        if not config:
            return {}

        # Validate numeric parameters
        if 'max_chunks' in config:
            config['max_chunks'] = max(1, min(20, int(config['max_chunks'])))

        if 'temperature' in config:
            config['temperature'] = max(0.0, min(2.0, float(config['temperature'])))

        if 'max_tokens' in config:
            config['max_tokens'] = max(100, min(4000, int(config['max_tokens'])))

        return config


