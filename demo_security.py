import os
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware import Middleware
from starlette.responses import JSONResponse


class SimpleIPRateLimitMiddleware(BaseHTTPMiddleware):
    """Lightweight in-memory IP rate limiter for public demo deployment."""

    def __init__(self, app):
        super().__init__(app)
        self.enabled = os.getenv("DEMO_RATE_LIMIT_ENABLED", "true").lower() == "true"
        self.max_requests = int(os.getenv("DEMO_RATE_LIMIT_MAX_REQUESTS", "30"))
        self.window_seconds = int(os.getenv("DEMO_RATE_LIMIT_WINDOW_SECONDS", "600"))
        self.requests = defaultdict(deque)

    def _get_client_ip(self, request):
        forwarded_for = request.headers.get("x-forwarded-for", "")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        if request.client and request.client.host:
            return request.client.host
        return "unknown"

    async def dispatch(self, request, call_next):
        if not self.enabled:
            return await call_next(request)

        if request.method in {"GET", "HEAD", "OPTIONS"}:
            return await call_next(request)

        path = request.url.path
        protected_prefixes = ("/queue/", "/call/", "/run/", "/api/")
        if not path.startswith(protected_prefixes):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        now = time.time()
        bucket = self.requests[client_ip]

        while bucket and bucket[0] <= now - self.window_seconds:
            bucket.popleft()

        if len(bucket) >= self.max_requests:
            return JSONResponse(
                {
                    "error": "Rate limit exceeded for this IP. Please wait before trying again.",
                    "retry_after_seconds": self.window_seconds,
                },
                status_code=429,
            )

        bucket.append(now)
        return await call_next(request)


def make_security_middleware():
    if os.getenv("DEMO_RATE_LIMIT_ENABLED", "true").lower() != "true":
        return []
    return [Middleware(SimpleIPRateLimitMiddleware)]
