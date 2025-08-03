import time
from collections import deque, defaultdict
from typing import Dict

from fastapi import Request, HTTPException


class RateLimiter:
    def __init__(self):
        self.requests: Dict[str, deque] = defaultdict(deque)

    def is_allowed(self, user_id: str, window_seconds: int = 60, max_requests: int = 20) -> Tuple[bool, int]:
        """Check if request is allowed under rate limit"""
        now = time.time()
        user_requests = self.requests[user_id]

        # Remove old requests outside the window
        while user_requests and user_requests[0] <= now - window_seconds:
            user_requests.popleft()

        # Check if under limit
        if len(user_requests) >= max_requests:
            retry_after = int(user_requests[0] + window_seconds - now + 1)
            return False, retry_after

        # Add current request
        user_requests.append(now)
        return True, 0


# Global rate limiter instance
rate_limiter = RateLimiter()


async def check_rate_limit(request: Request, user_id: str):
    """Middleware to check rate limits"""

    # Check per-minute limit
    allowed, retry_after = rate_limiter.is_allowed(user_id, 60, 20)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )

    # Check per-hour limit
    allowed, retry_after = rate_limiter.is_allowed(f"{user_id}_hour", 3600, 100)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Hourly rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )
