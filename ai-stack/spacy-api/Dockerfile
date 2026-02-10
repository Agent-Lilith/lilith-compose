# CPU-only, multi-language NER API (en, fr, de, nl, ru, ar, ms)
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install deps with uv (creates .venv)
COPY pyproject.toml .
RUN uv sync --no-install-project --no-dev \
    && uv pip install pip

# Download models into image (entrypoint copies to /data for host persistence)
RUN uv run python -m spacy download en_core_web_md \
    && uv run python -m spacy download fr_core_news_md \
    && uv run python -m spacy download de_core_news_md \
    && uv run python -m spacy download nl_core_news_md \
    && uv run python -m spacy download ru_core_news_md \
    && uv run python -m spacy download xx_ent_wiki_sm

COPY app/ ./app/
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
