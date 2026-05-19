# Modular RAG — Codebase Assistant

Ask natural-language questions about any codebase. Built with OpenAI + ChromaDB.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for a full design walkthrough.

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
├── chunkers/       # Split documents → Chunk (language-aware)
├── embedders/      # Text → vectors (OpenAI)
├── stores/         # Persist + search vectors (ChromaDB)
├── retrievers/     # Hybrid semantic + keyword search
├── rerankers/      # Cross-encoder reranking
├── context/        # Assemble prompt context from chunks
├── llm/            # Stream answers (OpenAI)
└── pipeline.py     # Public API: ingest() + query()
cli/
├── ingest.py       # CLI: index a directory
└── chat.py         # CLI: interactive Q&A
```

---

## Swapping Components

Every module satisfies a Protocol. Drop in replacements without touching the pipeline:

| Want | Do |
|------|----|
| Use Claude | Add `rag/llm/anthropic_llm.py` implementing `LLMProtocol` |
| Use Pinecone | Add `rag/stores/pinecone_store.py` implementing `VectorStoreProtocol` |
| Local embeddings | Add `rag/embedders/sentence_transformer_embedder.py` implementing `EmbedderProtocol` |
