# ==============================================================================
# Multi-stage production Dockerfile
# ==============================================================================

# --- Build Stage ---
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies to a local user directory
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Final Runtime Stage ---
FROM python:3.12-slim AS runner

WORKDIR /workspace

# Install curl for health check execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy python packages from builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy codebase
COPY . .

# Create non-root application user and setup permissions
RUN useradd -u 10001 -m appuser && \
    mkdir -p /workspace/data && \
    chown -R appuser:appuser /workspace

USER appuser

EXPOSE 8000
EXPOSE 8501

# Default environment configurations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000
