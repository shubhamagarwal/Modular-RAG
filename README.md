# Modular RAG — Codebase Assistant

Ask natural-language questions about any codebase. Built with OpenAI-compatible models (GitHub Copilot / GitHub Models), ChromaDB, and BM25.

---

## Documentation

| File | What it covers |
|------|---------------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | Full design — modules, protocols, extension points, design decisions |
| [FLOW.md](./FLOW.md) | Step-by-step code flow, file map, ASCII diagrams with log output at each stage |

---

## Quick Start

```bash
# 1. Install
pip3 install -e ".[dev]"

# 2. Configure
cp .env.example .env
# edit .env — set GITHUB_TOKEN to your GitHub PAT

# 3. Index a codebase (run once, or re-run when code changes)
python3 -m cli.ingest .                   # index current directory
python3 -m cli.ingest /path/to/repo      # or any other local repo

# 4. Chat (run after ingestion is complete)
python3 -m cli.chat
```

### Debug mode

```bash
python3 -m cli.ingest . --debug    # shows per-file and per-chunk detail
python3 -m cli.chat --debug        # shows BM25 scores, RRF fusion, reranker scores
```

---

## How It Works

Two pipelines run in sequence:

**Ingestion** — indexes your codebase once into ChromaDB + BM25:
```
Files → DirectoryLoader → CodeChunker → OpenAIEmbedder → ChromaStore
                                      └──────────────→ BM25Retriever (in-memory)
```

**Query** — retrieves and answers on every question:
```
Question → Embed → HybridRetriever (Semantic + BM25 → Weighted RRF)
                → CrossEncoderReranker → ContextBuilder → OpenAILLM → Answer
```

See [FLOW.md](./FLOW.md) for full diagrams with actual log output at each step.

---

## Running Tests

```bash
python3 -m pytest tests/unit -v
```

---

## Project Layout

```
rag/
├── loaders/        # Read files from disk → Document
├── chunkers/       # Split documents → Chunk (token-aware)
├── embedders/      # Text → vectors (GitHub Models API)
├── stores/         # Persist + search vectors (ChromaDB)
├── retrievers/     # BM25Retriever + HybridRetriever (weighted RRF)
├── rerankers/      # Cross-encoder reranking (ms-marco-MiniLM)
├── context/        # Assemble prompt context from chunks
├── llm/            # Stream answers (GitHub Models / gpt-4o-mini)
├── logger.py       # Centralised logging setup
└── pipeline.py     # Public API: ingest() + query()
cli/
├── ingest.py       # python3 -m cli.ingest <path> [--debug]
└── chat.py         # python3 -m cli.chat [--debug]
tests/unit/         # 12 unit tests, no external calls required
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `GITHUB_TOKEN` | — | GitHub PAT (needs `models:read` scope) |
| `API_BASE_URL` | `https://models.inference.ai.azure.com` | API endpoint |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model |
| `CHROMA_PERSIST_DIR` | `.chroma` | Vector store location |
| `CHUNK_SIZE` | `1200` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `10` | Candidates before reranking |
| `RERANK_TOP_K` | `4` | Final chunks sent to LLM |
| `CONTEXT_MAX_TOKENS` | `6000` | Max context fed to LLM |

---

## Swapping Components

Every module satisfies a Protocol — drop in replacements without touching the pipeline:

| Want | Do |
|------|----|
| Use Claude | Add `rag/llm/anthropic_llm.py` implementing `LLMProtocol` |
| Use Pinecone | Add `rag/stores/pinecone_store.py` implementing `VectorStoreProtocol` |
| Local embeddings | Add `rag/embedders/sentence_transformer_embedder.py` implementing `EmbedderProtocol` |
