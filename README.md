# Enterprise RAG Agent

Enterprise RAG Agent is a local, enterprise-style AI agent that answers questions from internal documents using Retrieval-Augmented Generation. It uses FastAPI for the API, LangGraph for orchestration, Ollama for local open-source LLM inference, ChromaDB for vector search, and SentenceTransformers for embeddings.

The first version is intentionally simple: run the stack locally, ingest markdown or text files from `data/docs`, and ask grounded questions through HTTP.

## Architecture

```text
User
  |
  v
FastAPI api service :8000
  |
  +--> LangGraph workflow
        |
        +--> classify_question_node
        +--> retrieve_context_node ---> ChromaDB service :8000 inside Docker
        +--> generate_answer_node ----> Ollama service :11434
        +--> fallback_answer_node

Host ports:
FastAPI  -> http://localhost:8000
Web UI   -> http://localhost:8000
ChromaDB -> http://localhost:8001
Ollama   -> http://localhost:11434
```

## Technologies

- Python 3.11
- FastAPI and Uvicorn
- LangGraph
- Ollama with `llama3.2:3b` by default
- ChromaDB
- SentenceTransformers using `sentence-transformers/all-MiniLM-L6-v2`
- Docker Compose
- Pytest, Ruff, mypy
- GitHub Actions

## Run Locally With Docker

From this directory:

```bash
docker compose up --build
```

In another terminal, pull the default Ollama model:

```bash
docker compose exec ollama ollama pull llama3.2:3b
```

The API container talks to Ollama and ChromaDB by Docker Compose service name:

- `http://ollama:11434`
- `chromadb:8000`

Open the web UI:

```text
http://localhost:8000
```

FastAPI's OpenAPI documentation is available at:

```text
http://localhost:8000/docs
```

## Run On A Home Lab Server

Use the standalone homelab compose file when another service already owns host port `8001`.
It maps:

- Web UI/API: `http://<server-ip>:8010`
- ChromaDB: `127.0.0.1:8011` on the server only
- Ollama: `127.0.0.1:11434` on the server only

On the server:

```bash
git clone <your-repo-url> enterprise-rag-agent
cd enterprise-rag-agent
cp .env.homelab.example .env
docker compose --env-file .env -f docker-compose.homelab.yml up -d --build
docker compose --env-file .env -f docker-compose.homelab.yml exec ollama ollama pull llama3.2:3b
curl -X POST http://localhost:8010/ingest
```

Open the UI from another machine on your LAN:

```text
http://<server-ip>:8010
```

If `8010`, `8011`, or `11434` are already used on the server, change the matching value in `.env`.

## Expose Through Traefik

Yes, use Traefik if you want public HTTPS access. Do not expose this app directly to the
internet without authentication; it currently has no login and no rate limiting.

The Traefik overlay only exposes the `api` service and removes the direct API host port.
ChromaDB and Ollama stay private.

First, find the Docker network used by your Traefik container:

```bash
docker inspect healthcare_traefik \
  --format '{{range $name, $_ := .NetworkSettings.Networks}}{{println $name}}{{end}}'
```

Edit `.env`:

```env
RAG_AGENT_HOST=rag.your-domain.com
TRAEFIK_NETWORK=<network-from-docker-inspect>
TRAEFIK_ENTRYPOINT=websecure
TRAEFIK_CERT_RESOLVER=letsencrypt
```

Create a DNS `A` record pointing `rag.your-domain.com` to your public IP, and make sure
ports `80` and `443` reach the Traefik server.

Start with the Traefik overlay:

```bash
docker compose --env-file .env \
  -f docker-compose.homelab.yml \
  -f docker-compose.traefik.yml \
  up -d --build
```

For public internet exposure, add Traefik Basic Auth. Generate a password hash on the
server:

```bash
sudo apt-get update
sudo apt-get install -y apache2-utils
htpasswd -nbB admin 'choose-a-strong-password' | sed -e 's/\$/\$\$/g'
```

Put the output in `.env`:

```env
TRAEFIK_BASIC_AUTH_USERS=admin:$$2y$$...
```

Then include the Basic Auth overlay:

```bash
docker compose --env-file .env \
  -f docker-compose.homelab.yml \
  -f docker-compose.traefik.yml \
  -f docker-compose.traefik.basicauth.yml \
  up -d --build
```

Then pull the model and ingest documents:

```bash
docker compose --env-file .env \
  -f docker-compose.homelab.yml \
  -f docker-compose.traefik.yml \
  exec ollama ollama pull llama3.2:3b

curl -X POST https://rag.your-domain.com/ingest
```

Open:

```text
https://rag.your-domain.com
```

## Health Check

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "services": {
    "api": "ok",
    "ollama": "ok",
    "chromadb": "ok"
  },
  "environment": "docker"
}
```

## Ingest Documents

The API ingests `.md` and `.txt` files from `data/docs`. The Docker Compose file mounts `./data/docs` into the API container at `/app/data/docs`.

```bash
curl -X POST http://localhost:8000/ingest
```

Example response:

```json
{
  "ingested_files": [
    "data_engineering_notes.md",
    "machine_learning_notes.md",
    "mlops_notes.md"
  ],
  "chunks_indexed": 6,
  "collection": "internal_docs"
}
```

## Ask Questions

Use the web UI at `http://localhost:8000`, or call the API directly:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How can I identify overfitting in a machine learning model?"}'
```

Example response:

```json
{
  "answer": "You can identify overfitting by comparing training and validation metrics...",
  "sources": ["machine_learning_notes.md"]
}
```

## LangGraph Workflow

The agent state contains:

- `question`
- `retrieved_context`
- `sources`
- `answer`
- `needs_retrieval`

The workflow is:

```text
classify_question_node
  |
  +-- needs retrieval --> retrieve_context_node
  |                         |
  |                         +-- context found --> generate_answer_node --> END
  |                         +-- no context ----> fallback_answer_node --> END
  |
  +-- no retrieval ----> generate_answer_node --> END
```

`classify_question_node` uses a simple heuristic to skip retrieval for greetings and assistant capability questions. Most knowledge questions use retrieval.

## How RAG Works Here

1. `/ingest` loads markdown and text files from `data/docs`.
2. Documents are split into overlapping chunks.
3. SentenceTransformers creates embeddings for each chunk.
4. Chunks, embeddings, metadata, and source filenames are stored in ChromaDB.
5. `/chat` embeds the user question.
6. ChromaDB returns the most relevant chunks.
7. The prompt sent to Ollama includes only the retrieved context and the question.
8. The API returns the model answer plus source filenames.

## Local Python Development

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
```

Copy the example environment file if you want local non-Docker settings:

```bash
cp .env.example .env
```

Run the API locally:

```bash
make run
```

For local non-Docker development, you still need Ollama and ChromaDB reachable at the URLs in `.env`.

## Tests, Linting, and Types

```bash
make test
make lint
make type-check
```

Or run everything manually:

```bash
ruff check .
mypy app
pytest
```

## GitHub Actions

The CI workflow at `.github/workflows/ci.yml` runs on pushes to `main` and pull requests. It installs the project with development dependencies, then runs:

- `ruff check .`
- `mypy app`
- `pytest`

## Configuration

Important environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `DOCS_PATH` | `data/docs` | Folder containing markdown and text files |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | Ollama model name |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `CHROMA_PORT` | `8001` | ChromaDB port for local host usage |
| `CHROMA_COLLECTION_NAME` | `internal_docs` | ChromaDB collection |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RETRIEVAL_TOP_K` | `4` | Number of chunks retrieved per question |

Docker Compose overrides the ChromaDB settings so the API uses `chromadb:8000` inside the Docker network.

## Troubleshooting

If `/health` shows Ollama as unavailable, pull and run the model:

```bash
docker compose exec ollama ollama pull llama3.2:3b
```

If `/chat` fails after startup, make sure the model pull completed. Ollama can be reachable before the requested model exists.

If `/ingest` fails, check that ChromaDB is running and that `data/docs` contains `.md` or `.txt` files.

If the first `/ingest` call is slow, the API may be downloading the SentenceTransformers embedding model. Later calls should be faster.

If port `8000` is already in use on your machine, change the FastAPI host port in `docker-compose.yml`.

## Future Improvements

- Replace ChromaDB with PostgreSQL + pgvector.
- Add authentication.
- Add a Streamlit or React frontend.
- Add a Jenkins pipeline.
- Add monitoring with Prometheus and Grafana.
- Add evaluation metrics for RAG quality.
- Add human-in-the-loop approval.
# ai_agents_orchestration
