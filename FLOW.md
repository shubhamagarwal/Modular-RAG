# Modular RAG — Code Flow & File Guide

---

## Project File Map

```
Modular-RAG/
│
├── .env                          ← your GITHUB_TOKEN lives here
├── rag/config.py                 ← loads .env, exposes `settings` object
│
├── rag/loaders/
│   ├── base.py                   ← Document dataclass + LoaderProtocol
│   ├── code_loader.py            ← reads .py .ts .js .go etc → Document
│   ├── markdown_loader.py        ← reads .md .mdx → Document
│   └── directory_loader.py       ← walks a folder, delegates to above loaders
│
├── rag/chunkers/
│   ├── base.py                   ← Chunk dataclass + ChunkerProtocol
│   └── code_chunker.py           ← splits Document text into overlapping Chunks
│
├── rag/embedders/
│   ├── base.py                   ← EmbedderProtocol
│   └── openai_embedder.py        ← calls GitHub Models API → list of vectors
│
├── rag/stores/
│   ├── base.py                   ← SearchResult dataclass + VectorStoreProtocol
│   └── chroma_store.py           ← saves/searches vectors in ChromaDB (.chroma/)
│
├── rag/retrievers/
│   ├── bm25_retriever.py         ← BM25Plus keyword index over all chunks (in-memory)
│   └── hybrid_retriever.py       ← fuses semantic + BM25 via weighted RRF
│
├── rag/rerankers/
│   ├── base.py                   ← RerankerProtocol
│   └── cross_encoder_reranker.py ← scores each chunk against the question
│
├── rag/context/
│   └── context_builder.py        ← formats top chunks as fenced code blocks
│
├── rag/llm/
│   ├── base.py                   ← LLMProtocol
│   └── openai_llm.py             ← streams answer from GitHub Models (gpt-4o-mini)
│
├── rag/pipeline.py               ← PUBLIC API: ingest() and query()
│
├── cli/
│   ├── ingest.py                 ← terminal command: python3 -m cli.ingest <path>
│   └── chat.py                   ← terminal command: python3 -m cli.chat
│
└── tests/unit/
    ├── test_chunker.py
    ├── test_context_builder.py
    ├── test_hybrid_retriever.py
    └── test_loader.py
```

---

## Pipeline 1 — Ingestion (run once)

**Entry point:** `cli/ingest.py` → calls `RAGPipeline.ingest(path)`

```
cli/ingest.py
    │
    │  python3 -m cli.ingest /your/repo
    ▼
rag/pipeline.py  →  ingest()
    │
    ├─► rag/loaders/directory_loader.py
    │       Walks the folder recursively.
    │       Skips: .git/, __pycache__/, node_modules/, .chroma/
    │       Respects .gitignore rules.
    │       Returns: list[Document]
    │           Document { content, file_path, language, repo_root }
    │
    ├─► rag/chunkers/code_chunker.py
    │       Splits each Document into overlapping windows.
    │       Uses tiktoken to count tokens (not characters).
    │       Default: 1200 tokens per chunk, 200 token overlap.
    │       Tracks start_line / end_line per chunk.
    │       Returns: list[Chunk]
    │
    ├─► rag/embedders/openai_embedder.py
    │       Sends chunks in batches of 100 to GitHub Models API.
    │       Model: text-embedding-3-small (1536-dim vectors).
    │       Returns: list[list[float]]  (one vector per chunk)
    │
    ├─► rag/stores/chroma_store.py
    │       Upserts chunks + vectors into ChromaDB.
    │       Stored on disk at: .chroma/
    │       Uses deterministic chunk IDs → re-ingesting is safe (no duplicates).
    │
    └─► rag/retrievers/bm25_retriever.py  (_rebuild_bm25)
            Builds an in-memory BM25Plus index from the full chunk list.
            Tokenizer splits on non-alphanumeric + camelCase boundaries:
                "authenticate_user" → ["authenticate", "user"]
            Rebuilt every time ingest() is called.

    ✓  Done → prints: "42 files → 318 chunks in 12.4s"
```

---

## Pipeline 2 — Query (run on every question)

**Entry point:** `cli/chat.py` → calls `RAGPipeline.query(question)`

```
cli/chat.py
    │
    │  User types: "How does authentication work?"
    ▼
rag/pipeline.py  →  query()
    │
    ├─► rag/embedders/openai_embedder.py
    │       Embeds the question into a vector.
    │       Same model as ingestion (text-embedding-3-small).
    │
    ├─► rag/retrievers/hybrid_retriever.py
    │       Runs TWO independent searches in parallel, then fuses:
    │
    │       Step 1 — Semantic search (ChromaDB):
    │           Queries ChromaDB with the question vector.
    │           Returns top-10 chunks by cosine similarity.
    │
    │       Step 2 — BM25 keyword search (bm25_retriever.py):
    │           Scores all chunks using BM25Plus TF-IDF algorithm.
    │           Catches exact identifiers: "verify_token", "AuthService" etc.
    │           Returns top-10 chunks by BM25 score.
    │
    │       Step 3 — Weighted RRF fusion:
    │           Merges both ranked lists using Reciprocal Rank Fusion.
    │           BM25 weight=1.5, Semantic weight=1.0
    │           (BM25 weighted higher: exact matches matter more in code)
    │           Final score = Σ weight / (rank + 60)
    │       Returns: list[SearchResult] (up to 10)
    │
    ├─► rag/rerankers/cross_encoder_reranker.py
    │       Downloads cross-encoder model on first run (ms-marco-MiniLM-L-6-v2).
    │       Scores each of the 10 candidates against the full question.
    │       Keeps top 4 most relevant chunks.
    │       Returns: list[SearchResult] (4)
    │
    ├─► rag/context/context_builder.py
    │       Formats the 4 chunks as fenced code blocks:
    │           ### src/auth/jwt.py  (lines 12–45)
    │           ```python
    │           def verify_token(token): ...
    │           ```
    │       Token budget: max 6000 tokens.
    │       Deduplicates chunks from same file + line range.
    │       Returns: str (the context string)
    │
    ├─► rag/llm/openai_llm.py
    │       Builds prompt:
    │           system: "You are a codebase assistant. Cite file + lines."
    │           user:   <context> + "Question: How does auth work?"
    │       Sends to GitHub Models (gpt-4o-mini), streams response.
    │       Yields: tokens one by one
    │
    └─► cli/chat.py
            Prints each token as it arrives (streaming).
            After answer: prints cited file paths + line ranges.
```

---

## Flow Diagram

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                         INGESTION  (run once)                             ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║   python3 -m cli.ingest /path/to/repo                                     ║
║         │                                                                 ║
║         ▼                                                                 ║
║   ┌──────────────────────────────────────────────────────────────────┐    ║
║   │ DirectoryLoader                                                  │    ║
║   │  • walks repo recursively                                        │    ║
║   │  • skips: .git/ __pycache__/ node_modules/ .chroma/              │    ║
║   │  • respects .gitignore rules                                     │    ║
║   │  LOG: "Scanning directory: /path/to/repo"                        │    ║
║   │  LOG: "42 files loaded, 3 skipped"                               │    ║
║   └─────────────────────────────┬────────────────────────────────────┘    ║
║                                 │  list[Document]                         ║
║                                 ▼                                         ║
║   ┌──────────────────────────────────────────────────────────────────┐    ║
║   │ CodeChunker                                                      │    ║
║   │  • splits each file into overlapping token windows               │    ║
║   │  • chunk_size=1200 tokens,  overlap=200 tokens                   │    ║
║   │  • tracks start_line / end_line per chunk                        │    ║
║   │  LOG: "Chunking 42 documents (chunk_size=1200, overlap=200)"     │    ║
║   │  LOG: "318 total chunks from 42 documents"                       │    ║
║   └──────────────────────────┬───────────────────────────────────────┘    ║
║                              │  list[Chunk]                               ║
║               ┌──────────────┴──────────────┐                            ║
║               ▼                             ▼                            ║
║   ┌────────────────────────┐   ┌─────────────────────────────────────┐   ║
║   │ OpenAIEmbedder         │   │ BM25Retriever  (_rebuild_bm25)      │   ║
║   │  • model:              │   │  • builds in-memory BM25Plus index  │   ║
║   │    text-embedding-3-   │   │  • tokenises: splits camelCase &    │   ║
║   │    small (1536-dim)    │   │    snake_case identifiers           │   ║
║   │  • batch size: 100     │   │    "verify_token" → [verify, token] │   ║
║   │  LOG: "Embedding 318   │   │  LOG: "Building BM25Plus index      │   ║
║   │  texts in 4 batches"   │   │        over 318 chunks"             │   ║
║   └───────────┬────────────┘   │  LOG: "BM25 index ready"            │   ║
║               │ list[vector]   └─────────────────────────────────────┘   ║
║               ▼                                                           ║
║   ┌────────────────────────┐                                              ║
║   │ ChromaStore            │                                              ║
║   │  • upserts by chunk_id │                                              ║
║   │    (idempotent)        │                                              ║
║   │  • persists to .chroma/│                                              ║
║   │  LOG: "Upserting 318   │                                              ║
║   │        chunks"         │                                              ║
║   └────────────────────────┘                                              ║
║                                                                           ║
║   ✓  Done → "42 files → 318 chunks in 12.4s"                             ║
╚═══════════════════════════════════════════════════════════════════════════╝


╔═══════════════════════════════════════════════════════════════════════════╗
║                      QUERY  (each question)                               ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║   python3 -m cli.chat                                                     ║
║   You: "Where is verify_token defined?"                                   ║
║         │                                                                 ║
║         ▼                                                                 ║
║   ┌──────────────────────────────────────────────────────────────────┐    ║
║   │ OpenAIEmbedder                                                   │    ║
║   │  • embeds question → 1536-dim query vector                       │    ║
║   │  LOG: "Embedding 1 texts in 1 batch(es)"                         │    ║
║   └─────────────────────────────┬────────────────────────────────────┘    ║
║                                 │  query vector                           ║
║                                 ▼                                         ║
║   ┌──────────────────────────────────────────────────────────────────┐    ║
║   │ HybridRetriever                                                  │    ║
║   │  LOG: "Mode: semantic + BM25"                                    │    ║
║   │                                                                  │    ║
║   │  query vector ──► ┌────────────────────────────────────────┐     │    ║
║   │                   │ Semantic Search  (ChromaDB)            │     │    ║
║   │                   │  cosine similarity over all vectors    │     │    ║
║   │                   │  LOG: "Semantic search — top 10"       │     │    ║
║   │                   │  LOG: "top score=0.8921"               │     │    ║
║   │                   └──────────────────┬─────────────────────┘     │    ║
║   │                                      │  ranked list A (10)       │    ║
║   │                                      │                           │    ║
║   │  raw query ──► ┌───────────────────────────────────────────┐     │    ║
║   │                │ BM25 Search  (BM25Retriever)              │     │    ║
║   │                │  LOG: "Query tokens: [verify, token]"     │     │    ║
║   │                │  LOG: "Threshold: 0.84 (10% of max=8.42)" │     │    ║
║   │                │  LOG: "[1] score=8.4210  src/auth/jwt.py  │     │    ║
║   │                │        lines 12-45"                       │     │    ║
║   │                │  LOG: "     preview: def verify_token..." │     │    ║
║   │                └──────────────────┬────────────────────────┘     │    ║
║   │                                   │  ranked list B (4)           │    ║
║   │                                   │                              │    ║
║   │              ┌────────────────────▼──────────────────────┐       │    ║
║   │              │ Weighted RRF Fusion                       │       │    ║
║   │              │  semantic weight = 1.0                    │       │    ║
║   │              │  BM25 weight     = 1.5                    │       │    ║
║   │              │  score = Σ weight / (rank + 60)           │       │    ║
║   │              │                                           │       │    ║
║   │              │  LOG: "Semantic candidates : 10"          │       │    ║
║   │              │  LOG: "BM25 candidates     : 4"           │       │    ║
║   │              │  LOG: "Unique after union  : 12"          │       │    ║
║   │              │  LOG: "[1] rrf=0.04139  sem=3  bm25=1     │       │    ║
║   │              │        src/auth/jwt.py:12-45"             │       │    ║
║   │              │  LOG: "[2] rrf=0.03500  sem=1  bm25=-     │       │    ║
║   │              │        src/auth/session.py:1-30"          │       │    ║
║   │              └────────────────────┬──────────────────────┘       │    ║
║   └───────────────────────────────────┼──────────────────────────────┘    ║
║                                       │  list[SearchResult] x10           ║
║                                       ▼                                   ║
║   ┌──────────────────────────────────────────────────────────────────┐    ║
║   │ CrossEncoderReranker                                             │    ║
║   │  • model: ms-marco-MiniLM-L-6-v2  (local, lazy-loaded)          │    ║
║   │  • scores every candidate against the full question             │    ║
║   │  • keeps top 4                                                  │    ║
║   │  LOG: "Reranking 10 candidates → keeping top 4"                 │    ║
║   │  LOG: "[1] score=9.21  src/auth/jwt.py lines 12-45"             │    ║
║   │  LOG: "[2] score=6.34  src/auth/middleware.py lines 5-38"       │    ║
║   └─────────────────────────────┬────────────────────────────────────┘    ║
║                                 │  list[SearchResult] x4                  ║
║                                 ▼                                         ║
║   ┌──────────────────────────────────────────────────────────────────┐    ║
║   │ ContextBuilder                                                   │    ║
║   │  • formats each chunk as a fenced code block:                   │    ║
║   │      ### src/auth/jwt.py  (lines 12–45)                         │    ║
║   │      ```python                                                   │    ║
║   │      def verify_token(token): ...                                │    ║
║   │      ```                                                         │    ║
║   │  • token budget: max 6000 tokens                                │    ║
║   │  • deduplicates same file + line range                          │    ║
║   │  LOG: "Context built — 4 blocks, 1823 total tokens"             │    ║
║   └─────────────────────────────┬────────────────────────────────────┘    ║
║                                 │  str context (≤6000 tokens)             ║
║                                 ▼                                         ║
║   ┌──────────────────────────────────────────────────────────────────┐    ║
║   │ OpenAILLM                                                        │    ║
║   │  • model: gpt-4o-mini  via GitHub Models API                    │    ║
║   │  • temperature=0  (deterministic)                               │    ║
║   │  • streaming=True                                               │    ║
║   │  LOG: "Sending request to models.inference.ai.azure.com"        │    ║
║   │  LOG: "Streaming response from LLM..."                          │    ║
║   │  LOG: "LLM stream complete — 142 tokens received"               │    ║
║   └─────────────────────────────┬────────────────────────────────────┘    ║
║                                 │  streamed tokens                        ║
║                                 ▼                                         ║
║   Answer printed token-by-token + cited file paths + line numbers         ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## Data Shapes at Each Step

| Step | Input | Output |
|------|-------|--------|
| DirectoryLoader | folder path | `list[Document]` |
| CodeChunker | `list[Document]` | `list[Chunk]` |
| OpenAIEmbedder (ingest) | `list[str]` (chunk text) | `list[list[float]]` (vectors) |
| ChromaStore.add | `list[Chunk]` + vectors | persisted to `.chroma/` |
| BM25Retriever (build) | `list[Chunk]` | in-memory BM25Plus index |
| OpenAIEmbedder (query) | `list[str]` (question) | `list[float]` (query vector) |
| ChromaStore.search | query vector | `list[SearchResult]` (10, semantic) |
| BM25Retriever.search | raw query string | `list[SearchResult]` (10, keyword) |
| HybridRetriever (RRF) | semantic list + BM25 list | `list[SearchResult]` (10, fused) |
| CrossEncoderReranker | question + fused results | `list[SearchResult]` (4) |
| ContextBuilder | `list[SearchResult]` | `str` (formatted context) |
| OpenAILLM | system + user + context | `AsyncIterator[str]` (tokens) |

---

## Config Reference (`rag/config.py`)

| Env Variable | Default | Controls |
|---|---|---|
| `GITHUB_TOKEN` | — | Auth for GitHub Models API |
| `API_BASE_URL` | `https://models.inference.ai.azure.com` | API endpoint |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model |
| `CHROMA_PERSIST_DIR` | `.chroma` | Where vectors are stored |
| `CHUNK_SIZE` | `1200` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `10` | Candidates before reranking |
| `RERANK_TOP_K` | `4` | Final chunks sent to LLM |
| `CONTEXT_MAX_TOKENS` | `6000` | Max context fed to LLM |
