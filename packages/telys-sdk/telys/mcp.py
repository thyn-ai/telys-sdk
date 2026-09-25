"""telys mcp — a minimal, dependency-free MCP server exposing Telys memory to any MCP client
(Claude Code/Desktop, Cursor, Cline, OpenAI Agents/ChatGPT — they all speak MCP).

Transport: newline-delimited JSON-RPC 2.0 over stdin/stdout (the MCP stdio transport). The client launches this
as a subprocess — **no port, no daemon** (fits the one-app/no-port model). It wraps the in-process engine
(`load_runtime`), so add/search are fully local and offline after `telys login`.

Full-replica surface: every capability of the regular `telys.Telys` / `Collection` SDK that makes sense for a
text-mode LLM caller is exposed as an MCP tool (CRUD, filtered queries, lexical search, compaction / IVF /
tuning, plus a repo auto-indexer for "map this codebase and let the LLM search it" workflows).

Auto-index: `telys_repo_search` lazily indexes the server's workspace on first call, then keeps the index
fresh on every subsequent search — walking the tree, comparing per-file (size, mtime) fingerprints, and
incrementally re-embedding only the files that changed. Respects `.gitignore` (via `git ls-files
--exclude-standard`) when the workspace is a git checkout; otherwise a small default `_SKIP_DIRS` set applies.

The tool operations live in `TelysMemory` and are reused by the `telys mem …` terminal commands, so the CLI
and the MCP plugin share one code path.

Audit: `telys mcp --monitor` appends every JSON-RPC request/response to `<memory_path>/mcp-monitor.jsonl`
(`--monitor-log PATH` overrides the location), so you can see exactly which tools the assistant host calls.
"""
from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from telys import __version__

PROTOCOL_VERSION = "2025-06-18"
_DEFAULT_DIM = 384  # AlgentaMultigramEmbedder

# ── auto-index defaults (safe, bounded — the LLM can override per-call) ─────────────────────────────────────
_DEFAULT_WINDOW_LINES = 48
_DEFAULT_WINDOW_OVERLAP = 8
# No per-file byte cap by default: "index the whole repo" means every text file, however large. Binaries are
# already excluded by the NUL-byte sniff in `_looks_binary`, and large text files are chunked like any other.
# `0` = unlimited; pass a positive integer per-call to bound it (e.g. on a memory-constrained host).
_DEFAULT_MAX_FILE_BYTES = 0
# Default: no ceiling on files or wall-clock — the auto-indexer walks the whole repo, no exceptions. The
# kernel-backed multigram embedder benchmarks at millions-of-lines-per-second, so full-repo indexing is
# table-stakes. `0` = unlimited (walker keeps going until no more files / no more time). The LLM can still
# opt into a ceiling per-call by passing a positive integer.
_DEFAULT_MAX_FILES = 0
_DEFAULT_MAX_SECONDS = 0
_INDEX_COLLECTION_DEFAULT = "repo_symbols"

# When there is no .gitignore to respect, fall back to skipping these obvious build/dep dirs.
_SKIP_DIRS = frozenset({
    ".git", "node_modules", ".venv", "__pycache__", "dist", "build",
    ".turbo", ".pixi", ".pixi-cache", ".pnpm-store", ".next", ".mypy_cache",
    ".pytest_cache", ".tox", ".gradle", "target",
})

_TEXTLIKE_CTYPE = re.compile(rb"[\x00]")  # any NUL byte → treat as binary


def default_memory_path() -> str:
    return os.environ.get("TELYS_MEMORY_PATH") or os.path.join(os.path.expanduser("~"), ".telys", "memory")


def default_workspace_path() -> str | None:
    """Server-side default workspace root for `telys_repo_search`. Env override wins; else None (LLM must pass
    an explicit `path` argument to telys_index_repo / telys_repo_search)."""
    return os.environ.get("TELYS_MCP_WORKSPACE") or None


# ── locking (concurrent MCP writes to the same memory path are exclusive) ───────────────────────────────────

@contextmanager
def _mcp_lock(memory_path: str) -> Iterator[None]:
    """fcntl.flock guard around mutating handlers. Two Cline windows sharing $TELYS_MEMORY_PATH otherwise race
    on col.save(). No-op on platforms without fcntl (Windows) — silently degrades to unlocked writes."""
    try:
        import fcntl  # POSIX only
    except ImportError:
        yield
        return
    lock_dir = memory_path
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, ".mcp.lock")
    fh = open(lock_path, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


class TelysMemory:
    """Thin, embedder-backed wrapper over the engine — shared by the MCP server and the `telys mem` CLI."""

    def __init__(self, path: str | None = None, workspace: str | None = None,
                 persistent_index: bool = False):
        self.path = path or default_memory_path()
        self.workspace = workspace or default_workspace_path()
        self.persistent_index = bool(persistent_index)
        self._engine = None
        self._embedder = None
        # per-repo fingerprint cache: repo_id → {relpath → (size, mtime_ns)}
        self._index_fp: dict[str, dict[str, tuple[int, int]]] = {}
        if self.persistent_index:
            self._load_fingerprints()

    def _embedder_obj(self):
        if self._embedder is None:
            from telys.embedding import AlgentaMultigramEmbedder  # runtime-backed (needs the installed engine)

            self._embedder = AlgentaMultigramEmbedder()
        return self._embedder

    def _eng(self):
        if self._engine is None:
            from telys import Telys

            self._engine = Telys(self.path, embedding_providers={"multigram": self._embedder_obj()})
        return self._engine

    def _collection(self, name: str, *, create: bool = False, partition_by: str = "scope",
                    filter_columns: list[str] | None = None):
        eng = self._eng()
        if name in eng.collections():
            return eng.open_collection(name, embedder=self._embedder_obj())
        if not create:
            raise ValueError(f"collection {name!r} does not exist (create it first)")
        cols = filter_columns if filter_columns is not None else [partition_by]
        return eng.create_collection(name, dim=self._embedder_obj().profile.dimension,
                                     partition_by=partition_by, embedder=self._embedder_obj(),
                                     filter_columns=cols)

    # ── existing tool ops (return plain dicts) ────────────────────────────────────────────────────────────
    def list_collections(self) -> dict:
        return {"collections": self._eng().collections()}

    def create_collection(self, name: str, partition_by: str = "scope") -> dict:
        self._collection(name, create=True, partition_by=partition_by)
        return {"created": name, "dim": self._embedder_obj().profile.dimension, "partition_by": partition_by}

    def add(self, collection: str, texts: list[str], ids: list[str] | None = None,
            metadata: list[dict] | None = None, partition_by: str = "scope") -> dict:
        if not texts:
            raise ValueError("texts must be non-empty")
        if ids is None:
            # Unique across repeated adds: bare m0..mN restarted at 0 on EVERY call, so the second
            # `telys mem add` / telys_add without explicit ids collided ("ids already exist").
            import uuid
            ids = [f"m-{uuid.uuid4().hex[:12]}" for _ in texts]
        if len(ids) != len(texts):
            raise ValueError("ids and texts length mismatch")
        meta = [dict(m) for m in (metadata or [{} for _ in texts])]
        for m in meta:
            m.setdefault(partition_by, "default")
        col = self._collection(collection, create=True, partition_by=partition_by)
        with _mcp_lock(self.path):
            col.add_texts([str(t) for t in texts], [str(i) for i in ids], meta)
            col.save()
        return {"added": len(texts), "collection": collection}

    def search(self, collection: str, query: str, top_k: int = 5, where: dict | None = None) -> dict:
        from telys import Eq

        col = self._collection(collection, create=False)
        w = _eq_from(where)
        res = col.search_text(str(query), top_k=int(top_k), where=w, with_metadata=True)
        return {"collection": collection, "query": query, "hits": _flatten_hits(res)}

    def stats(self, collection: str) -> dict:
        return {"collection": collection, "stats": self._collection(collection, create=False).stats()}

    # ── new tool ops: CRUD replicas of the SDK ────────────────────────────────────────────────────────────
    def upsert(self, collection: str, texts: list[str], ids: list[str] | None,
               metadata: list[dict] | None = None, partition_by: str = "scope") -> dict:
        if ids is None:
            # Explicit JSON null from a client is the same as omitting ids: upsert with generated ids == add.
            return self.add(collection, texts, None, metadata, partition_by)
        if len(ids) != len(texts):
            raise ValueError("ids and texts length mismatch")
        meta = [dict(m) for m in (metadata or [{} for _ in texts])]
        for m in meta:
            m.setdefault(partition_by, "default")
        col = self._collection(collection, create=True, partition_by=partition_by)
        with _mcp_lock(self.path):
            col.upsert_texts([str(t) for t in texts], [str(i) for i in ids], meta)
            col.save()
        return {"upserted": len(texts), "collection": collection}

    def update(self, collection: str, texts: list[str], ids: list[str] | None,
               metadata: list[dict] | None = None) -> dict:
        if ids is None:
            raise ValueError("update requires ids (strict by contract — use telys_upsert to insert)")
        if len(ids) != len(texts):
            raise ValueError("ids and texts length mismatch")
        col = self._collection(collection, create=False)
        meta = [dict(m) for m in (metadata or [{} for _ in texts])]
        for m in meta:
            # Same ergonomics as add/upsert: default the collection's partition key when metadata is omitted
            # (without it the engine's ingest raises a raw KeyError on m[key_name]).
            m.setdefault(col.key_name, "default")
        with _mcp_lock(self.path):
            col.update_texts([str(t) for t in texts], [str(i) for i in ids], meta)
            col.save()
        return {"updated": len(texts), "collection": collection}

    def delete(self, collection: str, ids: list[str]) -> dict:
        col = self._collection(collection, create=False)
        with _mcp_lock(self.path):
            col.delete([str(i) for i in ids])
            col.save()
        return {"deleted": len(ids), "collection": collection}

    def ids(self, collection: str, where: dict | None = None) -> dict:
        col = self._collection(collection, create=False)
        w = _eq_from(where)
        return {"collection": collection, "ids": col.ids(where=w)}

    def count(self, collection: str, where: dict | None = None) -> dict:
        col = self._collection(collection, create=False)
        w = _eq_from(where)
        return {"collection": collection, "count": len(col.ids(where=w))}

    def get(self, collection: str, ids: list[str], with_metadata: bool = True,
            with_text: bool = True) -> dict:
        """Exact row lookup by external id — no similarity search involved.

        Resolves external ids through the collection's id map and reads the stored filter columns straight
        out of the index via the runtime's `columns_for` row-metadata export (MVCC-aware).

        Text is *not* duplicated into the index — for auto-indexed repo rows the metadata carries
        `path` + `start_line` + `end_line`, and the exact source slice is re-read from disk on demand
        (`with_text=True`, the default). That keeps the vector store free of a second copy of the repo.
        """
        col = self._collection(collection, create=False)
        live = set(col.ids())
        id_map = col._id2int  # noqa: SLF001 — same-package accessor, mirrors engine-internal usage

        resolved: list[tuple[int, str, int]] = []  # (position, ext_id, internal_id)
        rows: list[dict[str, Any]] = []
        for pos, ext_id in enumerate(ids):
            rows.append({"id": ext_id, "found": ext_id in live})
            if ext_id in live and ext_id in id_map:
                resolved.append((pos, ext_id, int(id_map[ext_id])))

        if resolved and with_metadata:
            iids = [iid for _, _, iid in resolved]
            metas = col._rt.columns_for(col.idx, iids)  # noqa: SLF001 — runtime row-metadata export
            for (pos, _ext, _iid), meta in zip(resolved, metas):
                rows[pos]["metadata"] = meta
                if with_text:
                    text = _hydrate_text(meta, workspace=self.workspace)
                    if text is not None:
                        rows[pos]["text"] = text
        return {"collection": collection, "rows": rows}

    # ── new tool ops: query variants ──────────────────────────────────────────────────────────────────────
    def search_lexical(self, collection: str, query: str, top_k: int = 5,
                       where: dict | None = None, explain: bool = False) -> dict:
        col = self._collection(collection, create=False)
        w = _eq_from(where)
        res = col.search_text(str(query), top_k=int(top_k), where=w, mode="lexical",
                              explain=bool(explain), with_metadata=True)
        return {"collection": collection, "query": query, "hits": _flatten_hits(res)}

    # ── new tool ops: maintenance ─────────────────────────────────────────────────────────────────────────
    def compact(self, collection: str) -> dict:
        col = self._collection(collection, create=False)
        with _mcp_lock(self.path):
            col.compact()
            col.save()
        return {"compacted": collection}

    def build_ivf(self, collection: str, min_rows: int = 20000, target_recall: float = 0.98) -> dict:
        col = self._collection(collection, create=False)
        try:
            with _mcp_lock(self.path):
                col.build_ivf(int(min_rows), float(target_recall))
                col.save()
        except ModuleNotFoundError as e:
            if "faiss" in str(e):
                raise RuntimeError(
                    "IVF tuning needs the optional faiss dependency — install it into the telys environment "
                    "(pipx: `pipx inject telys faiss-cpu`; venv: `pip install faiss-cpu`). IVF only pays off "
                    "for large partitions (default threshold: 20000 rows); small collections use exact scans."
                ) from e
            raise
        return {"ivf_built": collection, "min_rows": int(min_rows), "target_recall": float(target_recall)}

    def build_lexical(self, collection: str, k1: float = 1.8, b: float = 1.0) -> dict:
        col = self._collection(collection, create=False)
        with _mcp_lock(self.path):
            col.build_lexical(float(k1), float(b))
            col.save()
        return {"lexical_built": collection, "k1": float(k1), "b": float(b)}

    def tune(self, collection: str, dry_run: bool | None = None) -> dict:
        col = self._collection(collection, create=False)
        plan = col.tune(dry_run=dry_run)
        plan_dict = dataclasses.asdict(plan) if dataclasses.is_dataclass(plan) else {"repr": repr(plan)}
        return {"collection": collection, "plan": plan_dict}

    # ── new tool ops: auto-index ──────────────────────────────────────────────────────────────────────────
    def index_repo(self, path: str | None = None, collection: str = _INDEX_COLLECTION_DEFAULT,
                   mode: str = "windowed", window_lines: int = _DEFAULT_WINDOW_LINES,
                   window_overlap_lines: int = _DEFAULT_WINDOW_OVERLAP,
                   max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
                   max_files: int = _DEFAULT_MAX_FILES,
                   max_seconds: int = _DEFAULT_MAX_SECONDS,
                   force: bool = False) -> dict:
        """Walk a repo dir → chunk each text file (windowed or whole-file) → ingest into a Telys collection.

        Incremental refresh: if the collection already has a fingerprint for this repo_id, walk the tree and
        embed ONLY the files whose (size, mtime_ns) changed. New files get ingested; removed files get their
        rows deleted. A no-change re-call is a fast no-op.

        Safety: respects `.gitignore` when the workspace is a git checkout (via
        `git ls-files -co --exclude-standard`). Otherwise falls back to a fixed skip list for common build /
        dep dirs. Rejects symlinks that resolve outside the repo root. Enforces max_files count + max_seconds
        wall-clock budget.
        """
        root = Path(path or self.workspace or os.getcwd()).resolve()
        if not root.is_dir():
            raise ValueError(f"workspace path not a directory: {root}")
        repo_id = _repo_id(root)
        allowed = _gitignore_allowlist(root)  # None → use _SKIP_DIRS defaults
        # Preflight: if the collection exists with a different partition_by, reject rather than reinterpret.
        eng = self._eng()
        if collection in eng.collections():
            existing = eng.open_collection(collection, embedder=self._embedder_obj())
            if existing.key_name != "repo_id":
                raise ValueError(
                    f"collection {collection!r} exists with partition_by={existing.key_name!r}; auto-index "
                    f"requires partition_by='repo_id'. Delete + re-create with a different collection name."
                )

        # `0` means "no limit" for both budgets. When set, the walker + reader honour them defensively.
        deadline = time.monotonic() + max_seconds if max_seconds > 0 else float("inf")
        new_fp: dict[str, tuple[int, int]] = {}
        new_texts: dict[str, str] = {}  # relpath → full file text (only for files that changed)
        walked = 0
        prior = self._index_fp.get(repo_id, {})
        for relpath, size, mtime_ns in _iter_repo_files(root, allowed=allowed, skip_dirs=_SKIP_DIRS,
                                                        max_file_bytes=max_file_bytes, deadline=deadline,
                                                        known=prior):
            walked += 1
            if max_files > 0 and walked > max_files:
                break
            new_fp[relpath] = (size, mtime_ns)

        old_fp = self._index_fp.get(repo_id, {})
        # `force` means "re-embed the repo regardless of the fingerprint" — so every tracked file counts as
        # changed. Deriving changed_paths from the fingerprint under force made it a silent no-op: the fast
        # path was skipped but there was still nothing to ingest.
        if force:
            changed_paths = list(new_fp)
        else:
            changed_paths = [p for p, sm in new_fp.items() if old_fp.get(p) != sm]
        removed_paths = [p for p in old_fp if p not in new_fp]

        # Fast path: no changes at all + collection exists.
        if not force and not changed_paths and not removed_paths and collection in eng.collections():
            return {"status": "hit", "collection": collection, "repo_id": repo_id,
                    "files_tracked": len(new_fp)}

        # Read the changed files' text (bounded — files were already size-checked in the walker).
        for rel in changed_paths:
            if time.monotonic() > deadline:
                break
            try:
                new_texts[rel] = (root / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

        col = self._collection(collection, create=True, partition_by="repo_id",
                               filter_columns=["repo_id", "path", "language",
                                               "start_line", "end_line"])

        # Build the (texts, ids, metas) triple OUTSIDE the lock so lock hold time is minimized.
        add_texts: list[str] = []
        add_ids: list[str] = []
        add_meta: list[dict] = []
        for rel, text in new_texts.items():
            if time.monotonic() > deadline:
                break
            chunks = _chunk_text(text, mode=mode, window_lines=window_lines,
                                 window_overlap_lines=window_overlap_lines)
            lang = _guess_language(rel)
            for chunk_idx, (start_line, end_line, chunk) in enumerate(chunks):
                if not chunk.strip():
                    continue
                add_texts.append(chunk)
                add_ids.append(f"{repo_id}:{rel}:{chunk_idx}")
                # NB: the chunk text is deliberately NOT stored as a filter column. Metadata values are
                # dictionary-encoded in the index (col_v2i / col_i2v), so a unique-per-row blob would blow
                # the dictionary up to one entry per chunk. `path` + line range is the pointer; the exact
                # source slice is re-read from disk on demand (see `_hydrate_text`).
                add_meta.append({
                    "repo_id": repo_id,
                    "path": rel,
                    "language": lang,
                    "chunk": chunk_idx,
                    "start_line": start_line,
                    "end_line": end_line,
                })

        with _mcp_lock(self.path):
            # Drop rows belonging to changed + removed files. Row ids are `{repo_id}:{relpath}:{chunk_idx}`,
            # so recover relpath by stripping the known repo_id prefix and the trailing chunk index — an
            # O(live) pass with a set membership test. (Prefix-matching every live id against every changed
            # path is O(live x changed): on a full rebuild of this repo that was 11912 x 1849 = 22M
            # startswith() calls and dominated the entire index at ~46s.)
            stale_paths = set(changed_paths) | set(removed_paths)
            stale_ids: list[str] = []
            if stale_paths:
                id_prefix = f"{repo_id}:"
                for row_id in col.ids():
                    if not row_id.startswith(id_prefix):
                        continue
                    rel = row_id[len(id_prefix):].rpartition(":")[0]
                    if rel in stale_paths:
                        stale_ids.append(row_id)
            if stale_ids:
                col.delete(stale_ids)
            if add_texts:
                col.upsert_texts(add_texts, add_ids, add_meta)
            col.save()

        self._index_fp[repo_id] = new_fp
        if self.persistent_index:
            self._save_fingerprints()

        return {
            "status": "built" if not old_fp else "refreshed",
            "collection": collection,
            "repo_id": repo_id,
            "workspace": str(root),
            "files_tracked": len(new_fp),
            "files_changed": len(changed_paths),
            "files_removed": len(removed_paths),
            "chunks_added": len(add_texts),
            "chunks_deleted": len(stale_ids) if changed_paths or removed_paths else 0,
        }

    def repo_search(self, query: str, top_k: int = 5, path_hint: str | None = None) -> dict:
        """Search the auto-indexed repo collection. Lazily indexes the server's workspace on first call
        (or when the fingerprint is stale). Guaranteed-fresh: every call re-fingerprints and incrementally
        re-embeds changed files before searching."""
        if not self.workspace:
            raise ValueError(
                "no workspace configured — pass `path` to telys_index_repo, set TELYS_MCP_WORKSPACE env, "
                "or launch `telys mcp --workspace <dir>`"
            )
        # Guaranteed-fresh: every repo_search runs index_repo first (fast on no-change).
        index_result = self.index_repo()
        from telys import Eq

        col = self._collection(_INDEX_COLLECTION_DEFAULT, create=False)
        where = None
        if path_hint:
            where = Eq("path", path_hint)
        else:
            root = Path(self.workspace).resolve()
            where = Eq("repo_id", _repo_id(root))
        res = col.search_text(str(query), top_k=int(top_k), where=where, with_metadata=True)
        return {"collection": _INDEX_COLLECTION_DEFAULT, "query": query,
                "index_status": index_result.get("status"),
                "hits": _flatten_hits(res, workspace=self.workspace, hydrate=True)}

    def workspace_info(self) -> dict:
        root = Path(self.workspace).resolve() if self.workspace else None
        return {
            "workspace": str(root) if root else None,
            "repo_id": _repo_id(root) if root else None,
            "collection": _INDEX_COLLECTION_DEFAULT,
            "files_tracked": len(self._index_fp.get(_repo_id(root), {})) if root else 0,
            "persistent_index": self.persistent_index,
        }

    # ── fingerprint persistence (opt-in via TelysMemory(persistent_index=True)) ──────────────────────────
    def _fingerprint_path(self) -> str:
        return os.path.join(self.path, ".telys_mcp_fingerprints.json")

    def _load_fingerprints(self) -> None:
        p = self._fingerprint_path()
        try:
            with open(p, encoding="utf-8") as fh:
                data = json.load(fh)
        except (FileNotFoundError, OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        # JSON has no tuples, so the persisted shape is {repo_id: {rel: [size, mtime_ns]}}.
        out: dict[str, dict[str, tuple[int, int]]] = {}
        for repo_id, files in data.items():
            if not isinstance(files, dict):
                continue
            rec: dict[str, tuple[int, int]] = {}
            for rel, sm in files.items():
                if isinstance(sm, list) and len(sm) == 2 and all(isinstance(x, int) for x in sm):
                    rec[rel] = (sm[0], sm[1])
            if rec:
                out[repo_id] = rec
        self._index_fp = out

    def _save_fingerprints(self) -> None:
        p = self._fingerprint_path()
        try:
            os.makedirs(self.path, exist_ok=True)
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                # Serialize tuples as lists (JSON has no tuple type).
                json.dump({r: {rel: list(sm) for rel, sm in files.items()}
                           for r, files in self._index_fp.items()}, fh)
            os.replace(tmp, p)
        except OSError:
            pass  # best-effort — a locked / full disk should not break MCP calls


# ── module-level helpers (auto-index walker, chunker, fingerprint) ──────────────────────────────────────────

def _eq_from(where: dict | None):
    """Build a single-key equality filter from a dict, or None. Rejects multi-key dicts."""
    if not where:
        return None
    if len(where) != 1:
        raise ValueError(f"where accepts a single equality key; got {len(where)} keys")
    from telys import Eq

    (k, v), = where.items()
    return Eq(k, v)


def _flatten_hits(res: Any, *, workspace: str | None = None, hydrate: bool = False) -> list[dict]:
    """Normalize a Collection.search_text() response to a list of {id, score?, metadata?, text?} dicts.

    When `hydrate` is set (repo-index hits), the exact source slice named by the row's `path` +
    `start_line`/`end_line` metadata is read back from disk — the index never stores a second copy of the
    repo, so this is where a hit becomes readable code for the LLM.
    """
    if not isinstance(res, dict):
        return []
    ids = res.get("ids") or []
    metas = res.get("metadata") or []
    scores = res.get("scores") or []
    hits: list[dict] = []
    for i, ext_id in enumerate(ids):
        row: dict[str, Any] = {"id": ext_id}
        if i < len(scores):
            row["score"] = scores[i]
        if i < len(metas):
            m = metas[i] if isinstance(metas[i], dict) else {}
            row["metadata"] = m
            if hydrate:
                text = _hydrate_text(m, workspace=workspace)
                if text is not None:
                    row["text"] = text
        hits.append(row)
    return hits


def _hydrate_text(meta: Any, *, workspace: str | None) -> str | None:
    """Re-read the exact source slice a repo-index row points at: `path` + `start_line`..`end_line`.

    Returns None when the row isn't a repo-index row, the workspace is unknown, the file is gone, or the
    path escapes the workspace root. Never raises — a hydration miss must not fail the surrounding query.
    """
    if not isinstance(meta, dict) or not workspace:
        return None
    rel = meta.get("path")
    if not isinstance(rel, str) or not rel:
        return None
    try:
        root = Path(workspace).resolve()
        target = (root / rel).resolve()
        if target != root and not str(target).startswith(str(root) + os.sep):
            return None  # path escape — refuse
        if not target.is_file():
            return None
        start = meta.get("start_line")
        end = meta.get("end_line")
        text = target.read_text(encoding="utf-8", errors="replace")
        if isinstance(start, int) and isinstance(end, int) and start >= 1 and end >= start:
            lines = text.splitlines(keepends=True)
            return "".join(lines[start - 1:end])
        return text
    except (OSError, ValueError):
        return None


def _repo_id(root: Path) -> str:
    """Stable per-workspace identifier (sha1 of resolved absolute path, first 16 hex chars)."""
    return hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:16]


def _gitignore_allowlist(root: Path) -> set[str] | None:
    """When root is a git checkout, ask git for the exact set of tracked + untracked-but-not-ignored files.
    Returns a set of relative paths (POSIX-slash) or None when git isn't usable — in which case the walker
    falls back to _SKIP_DIRS.

    Uses `git ls-files -co --exclude-standard -z`: tracked + others + respect .gitignore/exclude/global.
    """
    if not (root / ".git").exists():
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"],
            capture_output=True, timeout=15, check=False,
        )
        if proc.returncode != 0:
            return None
        return set(proc.stdout.decode("utf-8", errors="replace").split("\0")) - {""}
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _iter_repo_files(root: Path, *, allowed: set[str] | None, skip_dirs: frozenset,
                     max_file_bytes: int, deadline: float,
                     known: dict[str, tuple[int, int]] | None = None) -> Iterator[tuple[str, int, int]]:
    """Yield (relpath_posix, size_bytes, mtime_ns) for each ingestable file under root.

    - When `allowed` is set (git checkout), iterate exactly that list — trusting .gitignore.
    - When None, walk os.walk with `skip_dirs` pruning + a binary/size sniff.
    - Rejects symlinks whose real path escapes root.
    - Enforces the deadline: stops early if walk exceeds time budget.

    `known` is the previous run's {relpath: (size, mtime_ns)} fingerprint. A file whose (size, mtime_ns) is
    unchanged already passed the binary sniff last time, so the 8 KB read is skipped — that read, repeated
    across every file, is what makes an otherwise-cheap freshness walk dominate query latency.
    """
    known = known or {}
    if allowed is not None:
        for rel in sorted(allowed):
            if time.monotonic() > deadline:
                return
            candidate = root / rel
            try:
                if candidate.is_symlink():
                    real = candidate.resolve()
                    if not str(real).startswith(str(root.resolve()) + os.sep) and real != root.resolve():
                        continue
                if not candidate.is_file():
                    continue
                st = candidate.stat()
                if max_file_bytes > 0 and st.st_size > max_file_bytes:
                    continue
                key = rel.replace(os.sep, "/")
                if known.get(key) != (st.st_size, st.st_mtime_ns) and _looks_binary(candidate):
                    continue
                yield key, st.st_size, st.st_mtime_ns
            except (OSError, ValueError):
                continue
        return

    # Fallback: no .gitignore — use os.walk with the sensible-defaults skip list.
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if time.monotonic() > deadline:
            return
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        for name in filenames:
            if time.monotonic() > deadline:
                return
            if name.startswith("."):
                continue
            candidate = Path(dirpath) / name
            try:
                rel = candidate.relative_to(root)
            except ValueError:
                continue
            try:
                if candidate.is_symlink():
                    real = candidate.resolve()
                    if not str(real).startswith(str(root.resolve()) + os.sep) and real != root.resolve():
                        continue
                st = candidate.stat()
                if max_file_bytes > 0 and st.st_size > max_file_bytes:
                    continue
                key = str(rel).replace(os.sep, "/")
                if known.get(key) != (st.st_size, st.st_mtime_ns) and _looks_binary(candidate):
                    continue
                yield key, st.st_size, st.st_mtime_ns
            except (OSError, ValueError):
                continue


def _looks_binary(path: Path, probe_bytes: int = 8192) -> bool:
    """Return True if the first `probe_bytes` of the file contain a NUL byte (a cheap, reliable binary test)."""
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(probe_bytes)
        return b"\x00" in chunk
    except OSError:
        return True  # unreadable → treat as binary


def _chunk_text(text: str, *, mode: str, window_lines: int, window_overlap_lines: int
                ) -> list[tuple[int, int, str]]:
    """Chunk `text` into records for ingestion. Returns list of (start_line, end_line, body).

    mode="file"     — one record per file (line range covers the whole file).
    mode="windowed" — sliding line window (default). Overlap keeps context across chunk boundaries.
    """
    lines = text.splitlines(keepends=True)
    n = len(lines)
    if n == 0:
        return []
    if mode == "file":
        return [(1, n, "".join(lines))]
    if mode != "windowed":
        raise ValueError(f"unknown chunk mode: {mode!r}")
    step = max(1, window_lines - window_overlap_lines)
    out: list[tuple[int, int, str]] = []
    for start in range(0, n, step):
        end = min(n, start + window_lines)
        out.append((start + 1, end, "".join(lines[start:end])))
        if end == n:
            break
    return out


_LANG_BY_EXT = {
    ".py": "python", ".pyi": "python",
    ".ts": "typescript", ".tsx": "typescript", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".rs": "rust", ".go": "go", ".java": "java", ".kt": "kotlin",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".mojo": "mojo", ".zig": "zig", ".swift": "swift", ".m": "objective-c", ".mm": "objective-cpp",
    ".rb": "ruby", ".php": "php", ".pl": "perl", ".lua": "lua", ".r": "r", ".jl": "julia",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".fish": "shell",
    ".md": "markdown", ".mdx": "markdown", ".rst": "restructuredtext", ".txt": "text",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".ini": "ini", ".xml": "xml",
    ".html": "html", ".css": "css", ".scss": "scss",
    ".sql": "sql", ".proto": "protobuf", ".graphql": "graphql", ".dockerfile": "dockerfile",
}


def _guess_language(rel: str) -> str:
    name = os.path.basename(rel).lower()
    if name == "dockerfile" or name.endswith(".dockerfile"):
        return "dockerfile"
    if name.startswith("makefile"):
        return "make"
    ext = os.path.splitext(name)[1]
    return _LANG_BY_EXT.get(ext, "text")


# ── MCP tool schemas ────────────────────────────────────────────────────────────────────────────────────────

_WHERE_SCHEMA = {"type": "object", "maxProperties": 1,
                 "description": ("optional single-key equality filter, e.g. scope=project:acme "
                                 "(one key only)")}

TOOLS = [
    # ── existing 5 ────────────────────────────────────────────────────────────────────────────────────────
    {"name": "telys_search",
     "description": ("Semantic + lexical search over a Telys memory collection: the server embeds "
                     "`query` with its built-in on-device embedder and returns the best-matching "
                     "live rows as hits (id, score, metadata). Use to recall stored "
                     "notes/facts/decisions before answering. Choose this embedding-ranked path "
                     "when wording varies; choose telys_search_lexical when exact terms and BM25 "
                     "scoring matter. Required: collection, query. Optional: top_k (default 5) "
                     "and where — a single-key equality filter such as scope=project:acme (one "
                     "key only; MCP-created collections are partitioned by `scope`). Fails "
                     "(isError) when the collection does not exist — write first via telys_add or "
                     "telys_create_collection — or when `where` carries more than one key."),
     "inputSchema": {"type": "object", "required": ["collection", "query"], "properties": {
         "collection": {"type": "string",
                        "description": "name of an existing collection in the active store"},
         "query": {"type": "string",
                   "description": "text query; embedded on-device before searching"},
         "top_k": {"type": "integer", "default": 5,
                   "description": "maximum hits to return, ordered best-first"},
         "where": _WHERE_SCHEMA}},
     "annotations": {"readOnlyHint": True, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": False}},
    {"name": "telys_add",
     "description": ("Add text documents to a Telys memory collection: each text is embedded "
                     "on-device, inserted as a new row, and persisted; the collection is "
                     "auto-created (partition_by='scope') when absent. Use to store new memories. "
                     "Required: collection, texts. Optional: ids (one per text; auto-generated "
                     "when omitted — an id that already exists raises a conflict, so use "
                     "telys_upsert to replace) and metadata (one object per text; set a `scope` "
                     "value to route partitions and enable where-filtering later). Fails on empty "
                     "texts, ids/texts length mismatch, or duplicate ids. NOT idempotent: omitted "
                     "ids are freshly generated per call, so a repeat call inserts additional new "
                     "rows."),
     "inputSchema": {"type": "object", "required": ["collection", "texts"], "properties": {
         "collection": {"type": "string",
                        "description": "target collection; auto-created with "
                        "partition_by='scope' when absent"},
         "texts": {"type": "array", "items": {"type": "string"},
                   "description": "documents to embed on-device and insert; one new row per text"},
         "ids": {"type": "array", "items": {"type": "string"},
                 "description": "external ids, one per text positionally; auto-generated when "
                 "omitted"},
         "metadata": {"type": "array", "items": {"type": "object"},
                      "description": "one object per text positionally; set `scope` to route "
                      "partitions"}}},
     "annotations": {"readOnlyHint": False, "destructiveHint": False,
                     "idempotentHint": False, "openWorldHint": False}},
    {"name": "telys_create_collection",
     "description": ("Create a named Telys memory collection in the active store, wired to the "
                     "built-in embedder. Optional — telys_add auto-creates on first write — but "
                     "use it to fix the name and partition key up front. Required: name. "
                     "Optional: partition_by (default 'scope'; a single metadata key name — "
                     "compose compound keys into one string yourself). The new collection is "
                     "built but NOT persisted, so telys_list_collections shows it only after a "
                     "mutating tool (telys_add, telys_upsert, …) writes to it. Repeating with the "
                     "same name returns the existing collection unchanged."),
     "inputSchema": {"type": "object", "required": ["name"], "properties": {
         "name": {"type": "string", "description": "name for the new collection"},
         "partition_by": {"type": "string", "default": "scope",
                          "description": "metadata key used as the physical partition key"}}},
     "annotations": {"readOnlyHint": False, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": False}},
    {"name": "telys_list_collections",
     "description": ("List the Telys memory collections saved in the active store "
                     "($TELYS_MEMORY_PATH, default ~/.telys/memory). Use to discover what exists "
                     "before searching or writing. Takes no arguments and returns sorted names; "
                     "an empty store yields an empty list, not an error. Collections created but "
                     "never written to by a mutating tool do not appear — if names you expect are "
                     "missing, check the client points at the right store."),
     "inputSchema": {"type": "object", "properties": {}},
     "annotations": {"readOnlyHint": True, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": False}},
    {"name": "telys_stats",
     "description": ("Report stats for one Telys memory collection: name, partition key name, and "
                     "live external-id count, plus runtime-provided counters (row counts, "
                     "dimension, index state) that vary by runtime version. Use to confirm a "
                     "collection exists and gauge its size. Siblings: telys_list_collections "
                     "enumerates collections; telys_count returns cardinality only. Required: "
                     "collection. Fails when the collection does not exist."),
     "inputSchema": {"type": "object", "required": ["collection"], "properties": {
         "collection": {"type": "string",
                        "description": "name of an existing collection in the active store"}}},
     "annotations": {"readOnlyHint": True, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": False}},

    # ── CRUD replicas ─────────────────────────────────────────────────────────────────────────────────────
    {"name": "telys_upsert",
     "description": ("Add-or-replace text rows by external id: new ids are inserted; existing ids "
                     "are re-embedded and replaced as a versioned update (never a duplicate "
                     "physical row), then persisted. Use when the caller owns the ids and a "
                     "repeated call must converge instead of conflicting — this is the idempotent "
                     "write path. Required: collection, texts, ids (explicit JSON null is treated "
                     "as omitted = plain telys_add with generated ids). Optional: metadata (one "
                     "object per text) and partition_by (default 'scope'; the collection is "
                     "auto-created when absent). Fails on ids/texts length mismatch."),
     "inputSchema": {"type": "object", "required": ["collection", "texts", "ids"], "properties": {
         "collection": {"type": "string",
                        "description": "target collection; auto-created (with partition_by) when "
                        "absent"},
         "texts": {"type": "array", "items": {"type": "string"},
                   "description": "documents to embed on-device; one row per text"},
         "ids": {"type": "array", "items": {"type": "string"},
                 "description": "external ids, one per text; new ids insert, existing ids are "
                 "replaced"},
         "metadata": {"type": "array", "items": {"type": "object"},
                      "description": "one object per text positionally; the partition key is "
                      "defaulted when omitted"},
         "partition_by": {"type": "string", "default": "scope",
                          "description": "partition key used when auto-creating the collection"}}},
     "annotations": {"readOnlyHint": False, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": False}},
    {"name": "telys_update",
     "description": ("Replace the vectors/metadata of EXISTING ids by re-embedding new texts, "
                     "then persist. Use for deliberate corrections to known rows. Required: "
                     "collection, texts, ids — strict by contract: missing or null ids fail with "
                     "'update requires ids' (use telys_upsert to insert-or-replace instead). "
                     "Optional: metadata (one object per text; the collection's partition key is "
                     "defaulted when omitted). Fails when the collection does not exist or on "
                     "ids/texts length mismatch."),
     "inputSchema": {"type": "object", "required": ["collection", "texts", "ids"], "properties": {
         "collection": {"type": "string",
                        "description": "name of an existing collection in the active store"},
         "texts": {"type": "array", "items": {"type": "string"},
                   "description": "replacement texts, embedded on-device; one per id"},
         "ids": {"type": "array", "items": {"type": "string"},
                 "description": "external ids of EXISTING rows to replace (strict: required)"},
         "metadata": {"type": "array", "items": {"type": "object"},
                      "description": "one object per text positionally; the partition key is "
                      "defaulted when omitted"}}},
     "annotations": {"readOnlyHint": False, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": False}},
    {"name": "telys_delete",
     "description": ("Tombstone rows by external id so they no longer appear in queries or id "
                     "listings; the store is persisted and the space is physically reclaimed "
                     "later by telys_compact. Use to forget memories. Required: collection, ids. "
                     "Repeating the same delete is a no-op. Fails when the collection does not "
                     "exist."),
     "inputSchema": {"type": "object", "required": ["collection", "ids"], "properties": {
         "collection": {"type": "string",
                        "description": "name of an existing collection in the active store"},
         "ids": {"type": "array", "items": {"type": "string"},
                 "description": "external ids to tombstone"}}},
     "annotations": {"readOnlyHint": False, "destructiveHint": True,
                     "idempotentHint": True, "openWorldHint": False}},
    {"name": "telys_ids",
     "description": ("Return the live (non-tombstoned) external ids in a collection, optionally "
                     "scoped by a where filter — enumeration without ranking. Siblings: "
                     "telys_search returns ranked hits with scores for a query; telys_count "
                     "returns just the cardinality. Required: collection. Optional: where — a "
                     "single-key equality filter such as scope=project:acme (one key only). "
                     "Result: the full id list in a single response (no pagination); the order is "
                     "the engine's internal row order — neither sorted nor insertion order — so "
                     "sort client-side when a stable display order matters. Fails when the "
                     "collection does not exist or when `where` carries more than one key."),
     "inputSchema": {"type": "object", "required": ["collection"], "properties": {
         "collection": {"type": "string",
                        "description": "name of an existing collection in the active store"},
         "where": _WHERE_SCHEMA}},
     "annotations": {"readOnlyHint": True, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": False}},
    {"name": "telys_count",
     "description": ("Return the live row count (post-tombstone) for a collection — cardinality "
                     "only, optionally where-scoped. Siblings: telys_stats for aggregate stats "
                     "(partition key, live id count, runtime index counters); telys_ids to "
                     "enumerate the ids themselves. Required: collection — the name of an "
                     "existing collection in the active store; the call fails when it does not "
                     "exist (write first via telys_add or telys_create_collection). Optional: "
                     "where — a single-key equality filter (one key only); more than one key "
                     "fails."),
     "inputSchema": {"type": "object", "required": ["collection"], "properties": {
         "collection": {"type": "string",
                        "description": "name of an existing collection in the active store"},
         "where": _WHERE_SCHEMA}},
     "annotations": {"readOnlyHint": True, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": False}},
    {"name": "telys_get",
     "description": ("Exact row lookup by external id — no similarity search. Returns one entry "
                     "per requested id with a found flag plus stored metadata; for auto-indexed "
                     "repo rows the exact source slice is re-read from disk via the row's path + "
                     "line range. Use to fetch known rows or read the code behind a "
                     "telys_repo_search hit. Required: collection, ids. Optional: with_metadata "
                     "(default true) and with_text (default true; re-read the source slice for "
                     "repo-index rows). Unknown ids come back as found:false, not an error; fails "
                     "only when the collection does not exist."),
     "inputSchema": {"type": "object", "required": ["collection", "ids"], "properties": {
         "collection": {"type": "string",
                        "description": "name of an existing collection in the active store"},
         "ids": {"type": "array", "items": {"type": "string"},
                 "description": "external ids to look up exactly (no similarity search)"},
         "with_metadata": {"type": "boolean", "default": True,
                           "description": "include stored metadata for found rows"},
         "with_text": {"type": "boolean", "default": True,
                       "description": "re-read the source slice for repo-index rows"}}},
     "annotations": {"readOnlyHint": True, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": False}},

    # ── query variants ────────────────────────────────────────────────────────────────────────────────────
    {"name": "telys_search_lexical",
     "description": ("On-device BM25 lexical (keyword) search — exact-term matching for "
                     "identifiers and rare tokens where semantic search is fuzzy. Sibling: use "
                     "telys_search for embedding-ranked (vector) retrieval; this tool is the BM25 "
                     "path, no embedding involved. Requires the collection to have been created "
                     "with lexical=True AND telys_build_lexical to have run, otherwise the call "
                     "fails. Required: collection, query. Optional: top_k (default 5), where "
                     "(single-key equality filter), explain (default false; include per-hit score "
                     "explanations). Fails when the collection does not exist or the lexical "
                     "index was never built."),
     "inputSchema": {"type": "object", "required": ["collection", "query"], "properties": {
         "collection": {"type": "string",
                        "description": "name of an existing collection created with lexical=True"},
         "query": {"type": "string", "description": "keyword query for BM25 matching"},
         "top_k": {"type": "integer", "default": 5,
                   "description": "maximum hits to return, ordered best-first"},
         "where": _WHERE_SCHEMA,
         "explain": {"type": "boolean", "default": False,
                     "description": "include per-hit score explanations"}}},
     "annotations": {"readOnlyHint": True, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": False}},

    # ── maintenance ───────────────────────────────────────────────────────────────────────────────────────
    {"name": "telys_compact",
     "description": ("Flush tombstones and merge the delta segment into the base layout, "
                     "physically reclaiming space from deleted/updated rows; the store is "
                     "persisted. Use after large delete or update batches. Alternatives: "
                     "telys_delete to remove rows, telys_tune for index tuning. Required: "
                     "collection. Repeating with no pending changes is a no-op. Fails when the "
                     "collection does not exist."),
     "inputSchema": {"type": "object", "required": ["collection"], "properties": {
         "collection": {"type": "string",
                        "description": "name of an existing collection in the active store"}}},
     "annotations": {"readOnlyHint": False, "destructiveHint": True,
                     "idempotentHint": True, "openWorldHint": False}},
    {"name": "telys_build_ivf",
     "description": ("Build per-partition IVF indexes and calibrate nprobe to a recall floor — "
                     "approximate search for partitions that have outgrown exact scans. "
                     "Maintenance sibling of telys_compact (physically reclaim space from "
                     "deletes) and telys_tune (plan and optionally apply tuning automatically). "
                     "Required: collection. Optional: min_rows (default 20000; only partitions "
                     "with at least this many rows get an IVF index — smaller ones keep exact "
                     "scans) and target_recall (default 0.98). Fails when the collection does not "
                     "exist; needs the optional faiss dependency — the error names the fix (pipx: "
                     "`pipx inject telys faiss-cpu`; venv: `pip install faiss-cpu`)."),
     "inputSchema": {"type": "object", "required": ["collection"], "properties": {
         "collection": {"type": "string",
                        "description": "name of an existing collection in the active store"},
         "min_rows": {"type": "integer", "default": 20000,
                      "description": "only partitions with at least this many rows get an IVF "
                      "index (smaller stay exact)"},
         "target_recall": {"type": "number", "default": 0.98,
                           "description": "recall floor the nprobe calibration targets"}}},
     "annotations": {"readOnlyHint": False, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": False}},
    {"name": "telys_build_lexical",
     "description": ("Fit the BM25 lexical index over the collection's retained tokens and "
                     "persist it; the collection must have been created with lexical=True. This "
                     "builds the lexical BM25 index; telys_build_ivf builds the IVF vector index. "
                     "Run after bulk ingestion and before telys_search_lexical. Required: "
                     "collection. Optional: k1 (default 1.8; term-frequency saturation) and b "
                     "(default 1.0; document-length normalization). Fails when the collection "
                     "does not exist."),
     "inputSchema": {"type": "object", "required": ["collection"], "properties": {
         "collection": {"type": "string",
                        "description": "name of an existing collection created with lexical=True"},
         "k1": {"type": "number", "default": 1.8, "description": "BM25 term-frequency saturation"},
         "b": {"type": "number", "default": 1.0,
               "description": "BM25 document-length normalization"}}},
     "annotations": {"readOnlyHint": False, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": False}},
    {"name": "telys_tune",
     "description": ("Produce a TuningPlan for a collection via its Tuner, and optionally apply "
                     "it. Use to inspect or apply index/maintenance recommendations. Sibling: "
                     "telys_build_ivf builds the IVF index directly; this tool plans and "
                     "optionally applies tuning. Applying (dry_run=false) sets the default "
                     "target recall + exact↔quantized crossover and IVF-builds oversized "
                     "partitions (already-built ones are skipped) — instant settings plus a "
                     "faiss-requiring build that grows with partition size, applied live "
                     "in-process (this handler does not persist to disk) and not atomically, "
                     "so a mid-apply failure can leave earlier settings in place. Verify the "
                     "after-state with telys_stats. Required: collection. Optional: dry_run "
                     "(plan-only vs apply; values on the parameter). Fails when the "
                     "collection does not exist."),
     "inputSchema": {"type": "object", "required": ["collection"], "properties": {
         "collection": {"type": "string",
                        "description": "name of an existing collection in the active store"},
         "dry_run": {"type": "boolean",
                     "description": "true = return the plan only; false = plan + apply; "
                     "omit = the tuner's default"}}},
     "annotations": {"readOnlyHint": False, "destructiveHint": False,
                     "idempotentHint": False, "openWorldHint": False}},

    # ── auto-index (the LLM-facing repo scanner) ──────────────────────────────────────────────────────────
    {"name": "telys_index_repo",
     "description": ("Walk a repo directory — respecting .gitignore on git checkouts, else "
                     "skipping common build/dep dirs; binaries and symlinks escaping the root are "
                     "refused — chunk every text file, and ingest the chunks into a collection "
                     "for telys_repo_search. Incremental: re-calls re-fingerprint each file "
                     "(size, mtime), re-embed only changed files, tombstone removed ones, and "
                     "no-op fast when nothing changed. Routing: telys_repo_search queries the "
                     "index; telys_add/telys_upsert store manual memories. Returns status "
                     "(built/refreshed/hit), repo_id, and file/chunk counts. All arguments "
                     "optional: path (repo root; default: the server workspace, else CWD), "
                     "collection (default 'repo_symbols'), mode ('windowed' default or 'file'), "
                     "window_lines (48), window_overlap_lines (8), max_file_bytes / max_files / "
                     "max_seconds (0 = unlimited), force (default false; rebuild every file). "
                     "Fails when path is not a directory or the collection already exists with a "
                     "partition key other than repo_id."),
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string",
                  "description": "repo root (defaults to server workspace / CWD)"},
         "collection": {"type": "string", "default": _INDEX_COLLECTION_DEFAULT,
                        "description": "target collection for the repo chunks (auto-created with "
                        "partition_by='repo_id')"},
         "mode": {"type": "string", "enum": ["file", "windowed"], "default": "windowed",
                  "description": "'windowed' sliding line window or 'file' (one record per file)"},
         "window_lines": {"type": "integer", "default": _DEFAULT_WINDOW_LINES,
                          "description": "lines per chunk in windowed mode"},
         "window_overlap_lines": {"type": "integer", "default": _DEFAULT_WINDOW_OVERLAP,
                                  "description": "overlap lines between consecutive windows"},
         "max_file_bytes": {"type": "integer", "default": _DEFAULT_MAX_FILE_BYTES,
                            "description": "skip files larger than this many bytes; 0 = "
                            "unlimited"},
         "max_files": {"type": "integer", "default": _DEFAULT_MAX_FILES,
                       "description": "stop after walking this many files; 0 = unlimited"},
         "max_seconds": {"type": "integer", "default": _DEFAULT_MAX_SECONDS,
                         "description": "wall-clock budget in seconds; 0 = unlimited"},
         "force": {"type": "boolean", "default": False,
                   "description": "ignore the fingerprint cache and rebuild every file"}}},
     "annotations": {"readOnlyHint": False, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": False}},

    {"name": "telys_repo_search",
     "description": ("Search the auto-indexed repo collection (default 'repo_symbols') with the "
                     "on-device embedder. Every call first re-runs the incremental "
                     "telys_index_repo against the server workspace — cheap on no-change — so "
                     "hits are always fresh, and each hit carries the exact source slice re-read "
                     "from disk. Use for 'map this codebase and search it' workflows. Required: "
                     "query. Optional: top_k (default 5) and path_hint (exact repo-relative path "
                     "to scope the search). Fails when no workspace is configured — set "
                     "TELYS_MCP_WORKSPACE, launch `telys mcp --workspace <dir>`, or call "
                     "telys_index_repo with an explicit path first. Note: the freshness re-index "
                     "writes to the store, so this is not a read-only tool."),
     "inputSchema": {"type": "object", "required": ["query"], "properties": {
         "query": {"type": "string",
                   "description": "text query; embedded on-device before searching"},
         "top_k": {"type": "integer", "default": 5,
                   "description": "maximum hits to return, ordered best-first"},
         "path_hint": {"type": "string",
                       "description": "exact path (relative to repo root) to scope the search"}}},
     "annotations": {"readOnlyHint": False, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": False}},

    {"name": "telys_workspace_info",
     "description": ("Report the server's configured workspace: resolved path, repo_id, the "
                     "auto-index collection name, tracked file count, and whether the fingerprint "
                     "cache is persisted across restarts. Use to verify which repo "
                     "telys_repo_search will index before calling it. Takes no arguments; the "
                     "workspace fields are null when no workspace is configured — that is a "
                     "report, not an error."),
     "inputSchema": {"type": "object", "properties": {}},
     "annotations": {"readOnlyHint": True, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": False}},
]


_DISPATCH = {
    "telys_search": lambda m, a: m.search(**a),
    "telys_add": lambda m, a: m.add(**a),
    "telys_create_collection": lambda m, a: m.create_collection(**a),
    "telys_list_collections": lambda m, a: m.list_collections(),
    "telys_stats": lambda m, a: m.stats(**a),
    "telys_upsert": lambda m, a: m.upsert(**a),
    "telys_update": lambda m, a: m.update(**a),
    "telys_delete": lambda m, a: m.delete(**a),
    "telys_ids": lambda m, a: m.ids(**a),
    "telys_count": lambda m, a: m.count(**a),
    "telys_get": lambda m, a: m.get(**a),
    "telys_search_lexical": lambda m, a: m.search_lexical(**a),
    "telys_compact": lambda m, a: m.compact(**a),
    "telys_build_ivf": lambda m, a: m.build_ivf(**a),
    "telys_build_lexical": lambda m, a: m.build_lexical(**a),
    "telys_tune": lambda m, a: m.tune(**a),
    "telys_index_repo": lambda m, a: m.index_repo(**a),
    "telys_repo_search": lambda m, a: m.repo_search(**a),
    "telys_workspace_info": lambda m, a: m.workspace_info(),
}


class MCPServer:
    def __init__(self, memory: TelysMemory):
        self.memory = memory

    def handle(self, msg: dict) -> dict | None:
        method = msg.get("method")
        if method == "initialize":
            client_ver = (msg.get("params") or {}).get("protocolVersion") or PROTOCOL_VERSION
            return {"protocolVersion": client_ver, "capabilities": {"tools": {}},
                    "serverInfo": {"name": "telys", "version": __version__}}
        if method in ("notifications/initialized", "notifications/cancelled"):
            return None  # notification, no reply
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": TOOLS}
        if method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            fn = _DISPATCH.get(name)
            if fn is None:
                return {"content": [{"type": "text", "text": f"unknown tool: {name}"}], "isError": True}
            try:
                result = fn(self.memory, args)
                return {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}
            except Exception as exc:  # noqa: BLE001 — surface tool errors as MCP tool errors, not crashes
                return {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True}
        return {"__jsonrpc_error__": {"code": -32601, "message": f"method not found: {method}"}}

    def serve(self, stdin=None, stdout=None) -> int:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            result = self.handle(msg)
            msg_id = msg.get("id")
            if msg_id is None:
                continue  # notification (or a response we don't emit) — nothing to send back
            if isinstance(result, dict) and "__jsonrpc_error__" in result:
                out: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "error": result["__jsonrpc_error__"]}
            else:
                out = {"jsonrpc": "2.0", "id": msg_id, "result": result}
            stdout.write(json.dumps(out) + "\n")
            stdout.flush()
        return 0


# ── optional traffic monitor (`telys mcp --monitor`): audit what the assistant host actually calls ──────

def _monitor_record(log_path: str, direction: str, line: str) -> None:
    line = line.strip()
    if not line:
        return
    try:
        msg = json.loads(line)
    except ValueError:
        msg = {"raw": line}
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "dir": direction, "msg": msg}) + "\n")
    except OSError:
        pass  # best-effort — a log write failure must never break the server


def _monitor_stdin(log_path: str) -> Iterator[str]:
    for line in sys.stdin:
        _monitor_record(log_path, "in", line)
        yield line


class _MonitorStdout:
    """Tee every server->client line into the monitor log while passing writes through unchanged."""

    def __init__(self, log_path: str, inner: Any):
        self._log = log_path
        self._inner = inner

    def write(self, s: str) -> int:
        if s.strip():
            _monitor_record(self._log, "out", s)
        return self._inner.write(s)

    def flush(self) -> None:
        self._inner.flush()


def monitored_stdio(log_path: str) -> tuple[Any, Any]:
    """(stdin, stdout) pair for MCPServer.serve() that appends every JSON-RPC line to `log_path`."""
    return _monitor_stdin(log_path), _MonitorStdout(log_path, sys.stdout)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="telys mcp", description="Telys MCP stdio server")
    parser.add_argument("--path", default=None,
                        help="memory directory (default: ~/.telys/memory or $TELYS_MEMORY_PATH)")
    parser.add_argument("--workspace", default=None,
                        help=("repo root for the auto-indexer (default: $TELYS_MCP_WORKSPACE or none — "
                              "the LLM must then pass `path` to telys_index_repo / telys_repo_search)"))
    parser.add_argument("--persistent-index", action="store_true",
                        help=("persist the auto-index fingerprint cache to <memory_path>/.telys_mcp_"
                              "fingerprints.json so restarts skip re-embedding an unchanged repo"))
    parser.add_argument("--monitor", action="store_true",
                        help=("append every JSON-RPC request/response to <memory_path>/mcp-monitor.jsonl "
                              "so the assistant host's tool usage can be audited"))
    parser.add_argument("--monitor-log", default=None, metavar="PATH",
                        help="monitor log file (implies --monitor; default: <memory_path>/mcp-monitor.jsonl)")
    args = parser.parse_args(argv)
    memory = TelysMemory(args.path, workspace=args.workspace, persistent_index=args.persistent_index)
    log = args.monitor_log
    if args.monitor and not log:
        log = os.path.join(memory.path, "mcp-monitor.jsonl")
    stdin, stdout = monitored_stdio(log) if log else (None, None)
    return MCPServer(memory).serve(stdin=stdin, stdout=stdout)


if __name__ == "__main__":
    raise SystemExit(main())
