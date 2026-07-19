import time
from collections import defaultdict
from pathlib import Path
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import settings, logger

# Allowed formats based on ingestion loader router
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.ppt', '.html', '.htm', '.csv', '.json', '.txt', '.md'}

def validate_file_metadata(filename: str, file_size: int = 0):
    """
    Validates file extension and file size configurations before processing.
    """
    # 1. Validate Extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"File validation failed: Unsupported extension '{ext}' for file '{filename}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats are: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
        
    # 2. Validate Size (if size check is provided)
    if file_size > settings.MAX_UPLOAD_SIZE:
        logger.warning(f"File validation failed: File size {file_size} bytes exceeds limit of {settings.MAX_UPLOAD_SIZE} bytes")
        max_mb = settings.MAX_UPLOAD_SIZE / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum limit of {max_mb:.1f} MB."
        )

class InMemoryRateLimiter:
    """
    A sliding-window rate limiter utilizing in-memory IP request tracking.
    """
    def __init__(self, limit: int):
        self.limit = limit
        self.ip_records = defaultdict(list)

    def is_rate_limited(self, ip: str) -> bool:
        if self.limit <= 0:
            return False
            
        now = time.time()
        # Prune expired requests (older than 60 seconds)
        self.ip_records[ip] = [t for t in self.ip_records[ip] if now - t < 60]
        
        # Check against threshold
        if len(self.ip_records[ip]) >= self.limit:
            return True
            
        self.ip_records[ip].append(now)
        return False

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware class to enforce rate limits per client IP address.
    """
    def __init__(self, app, limit: int):
        super().__init__(app)
        self.limiter = InMemoryRateLimiter(limit)

    async def dispatch(self, request: Request, call_next):
        # Extract IP address supporting proxies (like Render)
        ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
        # Handle multiple proxies comma-delimited
        if "," in ip:
            ip = ip.split(",")[0].strip()
            
        if self.limiter.is_rate_limited(ip):
            logger.warning(f"Rate limit hit for IP: {ip} on route: {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Too Many Requests. IP rate limit exceeded. Please wait before retrying."}
            )
            
        return await call_next(request)
