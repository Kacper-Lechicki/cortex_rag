# Cortex

A CLI knowledge worker for technical articles. Scrape, ingest, query, visualize and evaluate — no LangChain, pure HuggingFace.

Point Cortex at any set of documentation URLs. Ask questions in natural language. Measure retrieval quality with MRR and Hit Rate. Iterate until your RAG is excellent.

## How it works

```
URL → scraper → chunker → ChromaDB (all-MiniLM-L6-v2 embeddings, local)
Question → retriever (+ optional query expansion) → HuggingFace zephyr-7b-beta → Answer
```

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` — runs locally, no API cost
- **Vector store**: ChromaDB with cosine distance, persisted on disk
- **Generation**: HuggingFace Inference API (free tier, rate limited)
- **Evaluation**: MRR and Hit Rate measured from a JSON evaluation set

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- HuggingFace account (free) with a read token (default), **or** an OpenAI API key (optional)

## Setup

```bash
git clone https://github.com/<your-username>/cortex-rag
cd cortex-rag
uv sync
cp .env.example .env
# Configure a generation provider and scraper allowlist in .env
```

`.env`:
```
# Required for scraping (SSRF-safe allowlist)
ALLOWED_DOMAINS="docs.trychroma.com, huggingface.co"

# Option A (default): HuggingFace
GENERATION_PROVIDER=hf-inference
HF_TOKEN=hf_your_token_here

# Option B: OpenAI
# GENERATION_PROVIDER=openai
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4o-mini
```

## Usage

```bash
# Launch the interactive menu (recommended)
uv run cortex

# Or run subcommands directly (script-friendly)

# Ingest articles
uv run cortex add https://huggingface.co/blog/rag https://docs.trychroma.com

# Ask questions
uv run cortex ask

# Run evaluation
uv run cortex eval --name "baseline"

# Visualize document map
uv run cortex viz docs

# Compare experiments
uv run cortex viz metrics

# Show stats
uv run cortex info
```

## Evaluation workflow

Edit `CHUNK_SIZE`, `TOP_K`, `USE_QUERY_EXPANSION` in `.env`, then:

```bash
uv run cortex clear && uv run cortex add <urls> && uv run cortex eval --name "experiment_name"
cortex viz metrics  # compare all runs side by side
```

## Project structure

```
src/cortex/
├── config.py       # pydantic-settings configuration
├── logging_utils.py # rotating file logger
├── scraper.py      # URL → clean text (httpx + BeautifulSoup)
├── chunker.py      # recursive character text splitting
├── store.py        # ChromaDB with cosine distance
├── retriever.py    # semantic search + Reciprocal Rank Fusion
├── generator.py    # HuggingFace InferenceClient + retry decorator
├── evaluator.py    # MRR, Hit Rate, NDCG from scratch
├── visualizer.py   # t-SNE document map + metrics bar chart
└── cli.py          # typer + Rich CLI
```

## License

MIT