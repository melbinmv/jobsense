# JobSense 🧠

> Real-time job market intelligence powered by a custom RAG pipeline.

JobSense searches live job listings and uses Claude AI to generate grounded, honest career insights with no hallucinations, no generic advice. Every answer is backed by real job data fetched from the Adzuna API.

---

## What it does

Ask JobSense anything about the job market:

- *"What data engineering roles are hiring in London right now?"*
- *"I know Python and Spark — what roles suit me?"*
- *"What salary should I expect as an MLOps engineer?"*

It fetches real listings, finds the most relevant ones using semantic search,
and hands them to Claude to generate a grounded answer with sources.

---

## Architecture

```
User Query
    │
    ▼
Embedder (all-MiniLM-L6-v2)     ← turns query into 384-dim vector
    │
    ▼
Vector Store (numpy)             ← cosine similarity search
    │
    ▼
Retriever                        ← deduplicates, ranks results
    │
    ▼
Synthesiser (Claude)             ← generates grounded answer
    │
    ▼
Answer + Sources
```

---

## Stack

| Layer | Technology |
|---|---|
| Data source | Adzuna Jobs API |
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 |
| Vector store | NumPy (custom, no external DB) |
| LLM | Claude claude-sonnet-4-6 (Anthropic) |
| MLOps | MLflow (experiment tracking) |
| API | FastAPI |
| CLI | Python argparse |

Built without LangChain or LlamaIndex — every component written from scratch.

---

## Project Structure

```
jobsense/
├── src/
│   ├── ingestion/
│   │   └── job_fetcher.py      # Adzuna API client
│   ├── pipeline/
│   │   ├── chunker.py          # splits job descriptions into chunks
│   │   ├── embedder.py         # sentence-transformers encoder + MLflow
│   │   └── vector_store.py     # numpy index — build / load / search
│   └── rag/
│       ├── retriever.py        # embed query → search → deduplicate
│       └── synthesiser.py      # context + Claude → grounded answer
├── api/
│   └── main.py                 # FastAPI — /health /index /query
├── pipeline.py                 # CLI entry point
├── config.yaml                 # chunk size, model, location, top_k
└── pyproject.toml
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/yourusername/jobsense.git
cd jobsense
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### 2. Add credentials

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
ADZUNA_APP_ID=your_app_id
ADZUNA_API_KEY=your_api_key
```

- Anthropic API key → [console.anthropic.com](https://console.anthropic.com)
- Adzuna credentials → [developer.adzuna.com](https://developer.adzuna.com) (free)

### 3. Build the index

```bash
# Fetch live jobs and build the vector index
python pipeline.py index

# Custom queries
python pipeline.py index \
  --queries "PySpark engineer" "MLOps engineer" "LLM engineer" \
  --max-per-query 20

# Re-index without hitting Adzuna again
python pipeline.py index --no-fetch
```

### 4. Query

```bash
python pipeline.py query "Python data engineer with Spark experience"
python pipeline.py query "entry level ML engineer London" --top-k 3
```

---

## API

Start the server:

```bash
uvicorn api.main:app --port 8000
```

Interactive docs at **http://localhost:8000/docs**

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check + index status |
| POST | `/index` | Fetch jobs and rebuild vector index |
| POST | `/query` | RAG query — returns answer + sources |
| GET | `/index/stats` | Detailed index statistics |

### Example

```bash
# Build index
curl -X POST http://localhost:8000/index \
     -H "Content-Type: application/json" \
     -d '{"queries": ["data engineer", "ML engineer"], "max_per_query": 15}'

# Query
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Python data engineer with Spark experience"}'
```

### Response

```json
{
  "answer": "Based on the current listings, two roles stand out...",
  "query": "Python data engineer with Spark experience",
  "model": "claude-sonnet-4-6",
  "input_tokens": 1198,
  "output_tokens": 312,
  "latency_secs": 8.95,
  "sources": [
    {
      "rank": 1,
      "title": "Data Engineer",
      "company": "Monzo",
      "location": "London, UK",
      "salary": "£70,000 – £90,000",
      "score": 0.8821,
      "url": "https://..."
    }
  ]
}
```

---

## MLflow Tracking

Every pipeline run is tracked automatically. Start the dashboard:

```bash
mlflow ui
# Open http://localhost:5000
```

Tracked experiments:
- `jobsense-embeddings` — encode time, chunks/sec, model name
- `jobsense-index` — index size, build time
- `jobsense-synthesis` — token usage, latency, chunks used

---

## Configuration

Edit `config.yaml` to tune the pipeline:

```yaml
chunk_size: 256       # words per chunk (matches model token limit)
chunk_overlap: 32     # words shared between adjacent chunks
model: all-MiniLM-L6-v2
top_k: 5              # results returned per query
location: London, UK  # default search location
```

---

