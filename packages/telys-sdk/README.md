# telys

<!-- mcp-name: io.github.thyn-ai/telys -->

**Public, thin SDK** for **Telys** — embedded, on-device memory & retrieval. In-process, **zero cloud
roundtrips** at query time.

This package contains only the developer-facing surface: the `Telys`/`Collection` facades, query/filter
types, the `EmbeddingProvider` interface, the `Tuner`/`TuningPlan` interfaces, a runtime loader, and the
`telys` CLI. **It contains no engine implementation** — the engine is a separate, closed, signed, on-device
runtime fetched by `telys runtime install` (the D-30 distribution decision: public SDK + closed runtime).

## Install

**As a CLI** (recommended — isolated env, on your PATH, not pinned to one Python):

```bash
pipx install telys
# …or the one-liner (installs via pipx):
curl -fsSL https://telys.ai/install.sh | sh

telys login          # sign in → free device license + signed runtime; fully offline thereafter
```

**As a library** (to `import telys` in your own project):

```bash
python -m venv .venv && . .venv/bin/activate
pip install telys
```

> Plain `pip install --user telys` works too, but pip may warn that its user-scripts dir isn't on your
> PATH (a macOS `--user` quirk) — `pipx` avoids that entirely. Runtime platforms: **macOS arm64, Linux
> x86_64/arm64** (Windows: run under **WSL2**).
```python
from telys import Telys
db = Telys("./memory")
col = db.create_collection("docs", dim=768, partition_by="tenant_id")   # dim is arbitrary — 384/768/1024/1536/3072…
col.add(vectors, ids=ids, metadata=metadata)          # bring your own vectors (embedding-agnostic, any dimension)
hits = col.search(qvec, where={"tenant_id": "acme"}, top_k=10, explain=True)
```

The **runtime is required for execution**; the **embedder is optional** — `col.add(vectors, …)` and
`col.add_texts(…)` both need the runtime, but only `*_texts` needs an embedder (bring your own via
`telys.embedding.CallableEmbedder`, or use the on-device bigram embedder).

For local development, install the runtime as a package instead of via the CLI:

```bash
pip install "telys[runtime]"   # or: pip install telys-runtime
```

## MCP server

[![MCP registry](https://img.shields.io/badge/MCP%20registry-io.github.thyn--ai%2Ftelys-blue)](https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.thyn-ai/telys)

`telys mcp` runs Telys as a [Model Context Protocol](https://modelcontextprotocol.io) server over stdio,
exposing **19 tools** — the full memory surface (CRUD, filtered queries, lexical search, compaction / IVF /
tuning, plus a repo auto-indexer) — to any MCP client (Claude Desktop/Code, Cursor, Codex, Qwen Code, …).
Everything is local and offline. Introspection (`initialize`/`tools/list`) needs no credentials; executing memory tools requires the one-time free `telys login` (device authorization, free community plan) — fully offline thereafter, no API key in the client config.

```bash
pipx install telys
telys login            # free device license + signed on-device runtime
telys runtime verify   # confirm the runtime the tools drive is present
telys mcp              # serve JSON-RPC 2.0 (protocol 2025-06-18) on stdin/stdout
```

You rarely start it by hand — `telys mcp install` writes the server entry into your clients — or add this
block to a client's MCP config yourself:

```json
{
  "mcpServers": {
    "telys": {
      "command": "telys",
      "args": ["mcp"]
    }
  }
}
```

Registry: [`io.github.thyn-ai/telys`](https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.thyn-ai/telys)
on the Official MCP Registry. Server source: [`packages/telys-sdk/telys/mcp.py`](packages/telys-sdk/telys/mcp.py).
MCP docs: [docs.telys.ai/mcp](https://docs.telys.ai/mcp/overview).

| Tool | What it does | Required arguments |
| --- | --- | --- |
| `telys_search` | Semantic + lexical search over a memory collection | `collection`, `query` |
| `telys_add` | Add text documents (collection auto-created if absent) | `collection`, `texts` |
| `telys_create_collection` | Create a collection (fix name + partition key up front) | `name` |
| `telys_list_collections` | List saved collections in the active store | — |
| `telys_stats` | Row counts and index state for one collection | `collection` |
| `telys_upsert` | Add-or-replace rows by id (the idempotent write) | `collection`, `texts`, `ids` |
| `telys_update` | Re-embed/replace EXISTING ids (strict: ids required) | `collection`, `texts`, `ids` |
| `telys_delete` | Tombstone rows by id | `collection`, `ids` |
| `telys_ids` | List live external ids, optionally `where`-scoped | `collection` |
| `telys_count` | Live row count, optionally `where`-scoped | `collection` |
| `telys_get` | Exact row lookup by id (re-reads repo source slices) | `collection`, `ids` |
| `telys_search_lexical` | On-device BM25 keyword search | `collection`, `query` |
| `telys_compact` | Flush tombstones, merge delta into the base layout | `collection` |
| `telys_build_ivf` | Build per-partition IVF indexes (recall-floor calibrated) | `collection` |
| `telys_build_lexical` | Fit the BM25 lexical index | `collection` |
| `telys_tune` | Produce (and optionally apply) a TuningPlan | `collection` |
| `telys_index_repo` | Walk + chunk + ingest a repo, incrementally refreshed | — |
| `telys_repo_search` | Search the auto-indexed repo (re-indexes first, always fresh) | `query` |
| `telys_workspace_info` | Report the configured workspace, repo_id and file count | — |

Every tool ships a rich description (what it does, when to use it, parameters with defaults, failure modes)
and static MCP annotations (`readOnlyHint` / `destructiveHint` / `idempotentHint` / `openWorldHint`) —
`openWorldHint: false` throughout: the server makes no network calls. Listing the tools needs **no
credentials and no runtime**: `initialize` + `tools/list` answer from the static tool registry; only actual
tool calls load the signed runtime lazily.

## Documentation

Full documentation lives at **[docs.telys.ai](https://docs.telys.ai)**:

- [Quickstart: your first collection](https://docs.telys.ai/quickstart-first-collection)
- [Working with data](https://docs.telys.ai/guides/create-a-collection) — add, upsert, search, filter, snapshot
- [Embeddings](https://docs.telys.ai/embeddings/choosing-an-embedder) — built-in lexical + bring-your-own via `CallableEmbedder`
- [CLI reference](https://docs.telys.ai/cli/overview) · [MCP integration](https://docs.telys.ai/mcp/overview) · [Self-hosting](https://docs.telys.ai/serve/overview)
- [SDK API index](https://docs.telys.ai/reference/sdk-api-index)

## Links

- Website: [telys.ai](https://telys.ai) · Pricing: [telys.ai/pricing](https://telys.ai/pricing)
- Account & billing: [accounts.thyn.ai](https://accounts.thyn.ai)
- Docs: [docs.telys.ai](https://docs.telys.ai) · PyPI: [pypi.org/project/telys](https://pypi.org/project/telys)
