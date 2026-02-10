FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (including building fasttext)
RUN uv sync --frozen --no-dev --no-install-project

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Copy Python and installed packages from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY app.py .
COPY download_model.py .

# Create models directory
RUN mkdir -p /models

EXPOSE 8000

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

CMD ["sh", "-c", "python download_model.py && uvicorn app:app --host 0.0.0.0 --port 8000"]

