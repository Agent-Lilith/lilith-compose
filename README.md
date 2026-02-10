# lilith-compose

Docker Compose stacks for the Lilith agent: AI/ML APIs and agent tooling (search, crawl, DB).

## Layout

| Stack | Contents |
|-------|----------|
| **ai-stack/** | vLLM (LLM), Whisper ASR, TEI embeddings, spaCy NER API, fastText langdetect API |
| **agent-stack/** | PostgreSQL (pgvector), SearXNG, FlareSolverr, Crawl4ai |

## Quick start

```bash
cp ai-stack/.env.example ai-stack/.env
cp agent-stack/.env.example agent-stack/.env

cd ai-stack && docker compose up -d
cd agent-stack && docker compose up -d
```

## Ports

| Stack | Service | Port |
|-------|---------|------|
| ai-stack | vLLM | 6001 |
| ai-stack | Whisper | 6002 |
| ai-stack | Embedding (TEI) | 6003 |
| ai-stack | spaCy | 6004 |
| ai-stack | fastText | 6005 |
| agent-stack | PostgreSQL | 6100 |
| agent-stack | SearXNG | 6101 |
| agent-stack | FlareSolverr | 6102 |
| agent-stack | Crawl4ai | 6103 |
