# Modular RAG — Architecture

A codebase assistant built on Retrieval-Augmented Generation (RAG). Ask natural-language questions about any Python/JS/TS project and get grounded, cited answers.

---

## Naive RAG vs Advanced RAG vs Modular RAG

### Naive RAG (the baseline)

```
Question → Embed → Vector Search → Top-K Chunks → LLM → Answer
```

That's it. Fixed pipeline, no flexibility. Problems:
- Bad retrieval = bad answer, no fallback
- Chunks are fixed-size text splits (cuts mid-function)
- No way to improve one stage without rewriting everything
- One vector store, one embedder, hardcoded

---

### Advanced RAG (better retrieval, same rigid structure)

Adds techniques **on top of** the naive pipeline but it's still one monolithic flow:

```
Question
  → Query Rewriting          ← pre-retrieval improvement
  → Hybrid Search (BM25 + semantic)
  → Reranking                ← post-retrieval improvement
  → Context compression
  → LLM → Answer
```

Better results, but the pipeline is still **fixed**. You can't swap the retriever, change the chunking strategy, or plug in a different vector store without modifying core code.

---

### Modular RAG (what we're building)

Each stage is an **independent, swappable module** behind a protocol:

```
[Loader] → [Chunker] → [Embedder] → [Store]
                                        ↓
[LLM] ← [ContextBuilder] ← [Reranker] ← [Retriever]
```

Every box satisfies a protocol (`EmbedderProtocol`, `VectorStoreProtocol`, etc.). You compose a pipeline from parts:

| Need | Swap |
|------|------|
| Use Claude instead of GPT | Plug in `AnthropicLLM` |
| Use Pinecone in prod | Plug in `PineconeStore` |
| Better chunking for Go code | Plug in `TreeSitterChunker` |
| No reranking for speed | Remove `CrossEncoderReranker` |

---

### Side-by-side summary

| | Naive RAG | Advanced RAG | Modular RAG |
|---|---|---|---|
| Pipeline | Fixed, linear | Fixed, enhanced | Composable modules |
| Query rewriting | No | Yes | Yes (optional module) |
| Chunking | Fixed-size text | Fixed-size text | Language-aware, swappable |
| Retrieval | Semantic only | Hybrid | Hybrid + pluggable |
| Reranking | No | Yes | Yes (optional, swappable) |
| Swap LLM | Rewrite code | Rewrite code | Swap one class |
| Swap vector store | Rewrite code | Rewrite code | Swap one class |
| Testability | Hard | Hard | Each module tested in isolation |

**TL;DR:** Naive RAG is a prototype. Advanced RAG is a smarter prototype. Modular RAG is a production architecture — same techniques as Advanced RAG but with clean separation so each piece is independently testable, replaceable, and configurable.

---

## How It Works (End-to-End)

```
┌─────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                   │
│                                                             │
│  Source Code  ──►  Loader  ──►  Chunker  ──►  Embedder     │
│  (files/dirs)                                  │            │
│                                                ▼            │
│                                          ChromaDB           │
│                                        (vector store)       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                       │
│                                                             │
│  User Question  ──►  Query Rewriter  ──►  Retriever         │
│                                               │             │
│                                        ┌──────▼──────┐      │
│                                        │  Semantic   │      │
│                                        │  Search     │      │
│                                        │ (ChromaDB)  │      │
│                                        └──────┬──────┘      │
│                                               │             │
│                                          Reranker           │
│                                               │             │
│                                       Context Builder       │
│                                               │             │
│                                        OpenAI LLM           │
│                                               │             │
│                                       Answer + Citations    │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Breakdown

### 1. Loader (`rag/loaders/`)

Reads source files from disk and normalises them into `Document` objects.

```
Document
  ├── content: str          # raw source text
  ├── metadata
  │     ├── file_path: str
  │     ├── language: str   # python | typescript | javascript | …
  │     └── repo_root: str
  └── id: str               # deterministic hash of path
```

Supported loaders:

| Loader | Handles |
|--------|---------|
| `CodeLoader` | `.py`, `.ts`, `.js`, `.go`, `.java`, `.rs` |
| `MarkdownLoader` | `.md`, `.mdx` |
| `TextLoader` | `.txt`, `.env.example`, `Makefile` |

The `DirectoryLoader` orchestrates them — walks a root, picks the right loader per extension, respects `.gitignore` patterns.

---

### 2. Chunker (`rag/chunkers/`)

Splits large files into overlapping windows that fit in embedding context.

**Strategy:** Language-aware chunking via tree-sitter (where available), falling back to recursive character splitting.

```
CodeChunker
  ├── chunk_size:    1 200 tokens   (fits most function definitions)
  ├── chunk_overlap:  200 tokens   (preserves context at boundaries)
  └── metadata appended per chunk:
        ├── chunk_index: int
        ├── start_line: int
        └── end_line: int
```

Why not fixed-size only? Function/class boundaries matter for code. A chunk that ends mid-function is noise to the LLM.

---

### 3. Embedder (`rag/embedders/`)

Converts text chunks into dense vectors.

```
OpenAIEmbedder
  ├── model:      text-embedding-3-small   (1 536 dims, cheap)
  │   (swap to)  text-embedding-3-large   (3 072 dims, higher recall)
  ├── batch_size: 100
  └── Implements: EmbedderProtocol
```

`EmbedderProtocol` is a simple `embed(texts: list[str]) -> list[list[float]]` interface — swap to a local model (e.g., `all-MiniLM-L6-v2` via sentence-transformers) without touching the rest.

---

### 4. Vector Store (`rag/stores/`)

Persists embeddings and enables similarity search.

```
ChromaStore
  ├── persist_directory: .chroma/
  ├── collection_name:   codebase
  └── Implements: VectorStoreProtocol
        ├── add(documents)
        ├── search(query_vector, k=10) → list[SearchResult]
        └── delete(ids)
```

ChromaDB runs in-process (no server required). The `.chroma/` directory is the sole artifact of the ingestion pipeline.

---

### 5. Retriever (`rag/retrievers/`)

Orchestrates search strategies on top of the vector store.

```
HybridRetriever
  ├── semantic_search(query, k=10)      # cosine similarity in embedding space
  ├── keyword_filter(query)            # optional metadata filter (file_path LIKE …)
  └── fuse_results(semantic, keyword)  # Reciprocal Rank Fusion
```

For a codebase assistant, semantic alone misses exact identifiers (`MyClassName`, `some_function`). Keyword filtering on `content` catches those.

---

### 6. Reranker (`rag/rerankers/`)

After retrieval returns 10 candidates, the reranker scores each against the question and keeps the top-k.

```
CrossEncoderReranker
  ├── model: cross-encoder/ms-marco-MiniLM-L-6-v2   (local, fast)
  ├── top_k: 4
  └── Implements: RerankerProtocol
```

Why not just take top-4 from semantic search directly? Embedding models optimise for broad recall; a cross-encoder re-scores with the query in context and improves precision significantly.

---

### 7. Context Builder (`rag/context/`)

Assembles the final prompt context from reranked chunks.

```
ContextBuilder
  ├── max_tokens:  6 000   (leaves room for system prompt + answer)
  ├── format:      fenced code blocks with file path + line range header
  └── dedup:       drop chunks from same file/line range if overlapping
```

Output example:
```
### src/auth/jwt.py  (lines 12–45)
```python
def verify_token(token: str) -> dict:
    ...
```

---

### 8. LLM (`rag/llm/`)

Sends the assembled prompt to OpenAI and streams the response.

```
OpenAILLM
  ├── model:        gpt-4o-mini   (default, cheap)
  │   (swap to)    gpt-4o         (better reasoning)
  ├── temperature:  0              (deterministic for Q&A)
  ├── streaming:    True
  └── Implements: LLMProtocol
        └── generate(system, user, context) → AsyncIterator[str]
```

System prompt is fixed:
> "You are a codebase assistant. Answer questions using only the provided code snippets. Always cite the file path and line numbers."

---

### 9. Pipeline (`rag/pipeline.py`)

Wires everything together into two public entry points:

```python
class RAGPipeline:
    async def ingest(self, path: str | Path) -> IngestResult
    async def query(self, question: str) -> AsyncIterator[AnswerChunk]
```

`IngestResult` carries `files_processed`, `chunks_added`, `duration_s`.  
`AnswerChunk` carries `text`, `citations: list[Citation]`.

---

## Directory Layout

```
Modular-RAG/
├── rag/
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── base.py          # Document, LoaderProtocol
│   │   ├── code_loader.py
│   │   ├── markdown_loader.py
│   │   └── directory_loader.py
│   ├── chunkers/
│   │   ├── __init__.py
│   │   ├── base.py          # ChunkerProtocol
│   │   └── code_chunker.py
│   ├── embedders/
│   │   ├── __init__.py
│   │   ├── base.py          # EmbedderProtocol
│   │   └── openai_embedder.py
│   ├── stores/
│   │   ├── __init__.py
│   │   ├── base.py          # VectorStoreProtocol, SearchResult
│   │   └── chroma_store.py
│   ├── retrievers/
│   │   ├── __init__.py
│   │   └── hybrid_retriever.py
│   ├── rerankers/
│   │   ├── __init__.py
│   │   ├── base.py          # RerankerProtocol
│   │   └── cross_encoder_reranker.py
│   ├── context/
│   │   ├── __init__.py
│   │   └── context_builder.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py          # LLMProtocol
│   │   └── openai_llm.py
│   ├── pipeline.py          # RAGPipeline (public API)
│   └── config.py            # Settings (pydantic-settings)
├── cli/
│   ├── ingest.py            # `python -m cli.ingest <path>`
│   └── chat.py              # `python -m cli.chat`
├── tests/
│   ├── unit/
│   └── integration/
├── .env.example
├── pyproject.toml
├── ARCHITECTURE.md          # ← this file
└── README.md
```

---

## Configuration (`rag/config.py`)

All settings via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model |
| `CHROMA_PERSIST_DIR` | `.chroma` | Where ChromaDB writes to disk |
| `CHROMA_COLLECTION` | `codebase` | Collection name |
| `CHUNK_SIZE` | `1200` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `10` | Candidates before reranking |
| `RERANK_TOP_K` | `4` | Final chunks sent to LLM |
| `CONTEXT_MAX_TOKENS` | `6000` | Max context tokens |

---

## Data Flow (Detailed)

### Ingestion

```
DirectoryLoader.load(root)
  │  walks files, dispatches to CodeLoader / MarkdownLoader
  ▼
list[Document]
  │
CodeChunker.chunk(documents)
  │  splits by AST node boundaries, adds line metadata
  ▼
list[Chunk]
  │
OpenAIEmbedder.embed(chunks)          # batched API calls
  │  returns list[list[float]]
  ▼
ChromaStore.add(chunks, embeddings)   # upsert by deterministic ID
  │  persists to .chroma/
  ▼
IngestResult(files=N, chunks=M, duration=T)
```

### Query

```
User question: "How does authentication work?"
  │
QueryRewriter.rewrite(question)
  │  expands acronyms, adds synonyms for better recall
  ▼
OpenAIEmbedder.embed([rewritten_query])
  │
HybridRetriever.search(query_vector, raw_query)
  │  semantic k=10  +  keyword filter
  ▼
list[SearchResult]  (up to 10)
  │
CrossEncoderReranker.rerank(question, results)
  │  scores each with cross-encoder, keeps top 4
  ▼
list[SearchResult]  (4)
  │
ContextBuilder.build(results)
  │  formats as fenced code blocks with headers
  ▼
str  (context)
  │
OpenAILLM.generate(system_prompt, question, context)
  │  streams tokens
  ▼
AsyncIterator[AnswerChunk]  →  CLI / API
```

---

## Extension Points

| Want to… | Change this |
|----------|-------------|
| Use Anthropic Claude | Implement `LLMProtocol` in `rag/llm/anthropic_llm.py` |
| Use Pinecone | Implement `VectorStoreProtocol` in `rag/stores/pinecone_store.py` |
| Add BM25 keyword search | Add `BM25Retriever` alongside `HybridRetriever` |
| Expose as REST API | Add `api/` with FastAPI, call `RAGPipeline.query()` |
| Add chat history | Pass previous Q&A turns to `ContextBuilder` |
| Index GitHub repo | Add `GitHubLoader` that clones then delegates to `DirectoryLoader` |

---

## Key Design Decisions

**Why protocols over abstract base classes?** Python's structural subtyping means you can pass any object that satisfies the interface — easier to mock in tests.

**Why deterministic chunk IDs?** Re-ingesting the same file is idempotent. ChromaDB upserts by ID so you don't accumulate duplicates on re-index.

**Why reranking?** Embeddings compress semantics — two chunks can be close in vector space but one is far more relevant to a specific question. A cross-encoder rescores with full query awareness and consistently improves precision@4.

**Why `gpt-4o-mini` as default?** For codebase Q&A with a tight, well-formed context, a smaller model is usually sufficient. The bottleneck is retrieval quality, not generation quality.
