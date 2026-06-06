# Gnovis

*From Proto-Indo-European \*gneh₃- (to know) — "to have known"*

Persistent memory system for LLMs. SQLite + FTS5 + vector search, exposed via [Model Context Protocol (MCP)](https://modelcontextprotocol.io).

## Features

- **Hybrid retrieval** — FTS5 keyword search + 768-dim vector KNN, fused with Reciprocal Rank Fusion (RRF)
- **Multi-model provenance** — every memory tracks which model created it; queries return results partitioned by ownership with attribution
- **Challenge/debate system** — models can challenge each other's memories, debate, and resolve disagreements
- **Temporal filtering** — `before`/`after` date params for time-bounded queries ("what happened last week")
- **Abstention detection** — `min_similarity` threshold returns empty results when nothing relevant exists, instead of guessing
- **Host-aware** — memories can be scoped to specific machines (e.g., NAS config vs desktop config)
- **ACT-R scoring** — recency decay + access frequency, modeled on human memory research
- **Zero cloud dependencies** — everything runs locally on SQLite + Ollama

## Benchmark

LongMemEval (500 questions, ICLR 2025):

| System | R@5 | Notes |
|--------|-----|-------|
| **Gnovis** | **0.970** | Raw hybrid retrieval, no reranking |
| MemPalace | 0.966 | Raw (ChromaDB + SQLite) |
| MemPalace + Haiku | 1.000 | With LLM reranking |

Benchmark script and results in `benchmarks/longmemeval/`.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) with `nomic-embed-text` model (for vector embeddings)
- SQLite with FTS5 support (standard on most systems)

## Install

```bash
git clone https://github.com/dev-ben-c/gnovis.git
cd gnovis
./deploy.sh
```

Or manually:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[vectors,dev]"
ollama pull nomic-embed-text
```

## Usage

### With Claude Code (MCP stdio)

```bash
claude mcp add gnovis -- /path/to/gnovis/venv/bin/python -m engram.server
```

### As HTTP/SSE server (for remote access)

```bash
# Start the server
source venv/bin/activate
python -m engram.server --transport sse --host 0.0.0.0 --port 8093

# Connect from Claude Code
claude mcp add gnovis --transport sse http://your-host:8093/sse
```

### Health check

```bash
curl http://localhost:8093/health
# {"status":"ok","db":"/home/user/.engram/memory.db"}
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `remember` | Store a new memory (fact, episode, preference, or diary) |
| `recall` | Hybrid search with temporal filtering and abstention |
| `forget` | Delete a memory by ID or category+key |
| `update` | Modify fields of an existing memory |
| `get_context` | Quick snapshot of recent facts and preferences |
| `history` | View change history for a memory |
| `challenge` | Challenge another model's memory |
| `debate` | Multi-round debate on a challenged memory |
| `resolve_challenge` | Accept, reject, or merge a challenge |
| `find_disagreements` | Find where models disagree |
| `relate` | Create entity relationships |
| `get_entity` | Query entity relationships |
| `list_categories` | Browse memory organization |
| `list_models` | See which models have contributed |
| `stats` | Database statistics |
| `backfill_embeddings` | Generate vectors for unembedded memories |

### Recall parameters

```
recall(
  query: str,              # Natural language search
  category: str,           # Filter by category
  memory_type: str,        # "fact", "episode", "preference", "diary"
  tags: [str],             # Filter by tags
  limit: int = 10,         # Max results
  caller_model: str,       # Your model ID (enables provenance)
  scope: "all" | "own",    # All models or just yours
  host: str,               # Filter to specific host
  caller_host: str,        # Boost matching host
  before: str,             # ISO 8601 — memories created before this date
  after: str,              # ISO 8601 — memories created after this date
  min_similarity: float,   # L2 distance threshold for abstention
)
```

## Database

Default location: `~/.engram/memory.db`

Schema auto-migrates on startup. Current version: v5.

## Tests

```bash
source venv/bin/activate
pytest tests/ -q
```

## License

MIT
