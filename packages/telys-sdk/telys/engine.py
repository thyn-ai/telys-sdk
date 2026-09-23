"""Telys — the embedded product surface (public SDK facade).

The developer-facing API: an engine rooted at a directory, holding named collections. Each collection is a
filtered vector index (by physical layout) + an optional pluggable embedder, with external ids (strings or
ints), metadata-driven partition keys + filter columns (incl. off-key filters that also see delta rows),
explicit add/upsert/update/delete semantics, IVF for oversized partitions, an optional embedding-provider
registry, and durable persistence.

This module talks ONLY to a RuntimeHandle (telys.runtime.load_runtime) — it imports no engine implementation;
the closed runtime is loaded the first time you create/open a collection (D-30).

    from telys import Telys, scope_key
    from telys.embedding import AlgentaBigramEmbedder
    eng = Telys("./codna-memory", embedding_providers={"bigram": AlgentaBigramEmbedder()})
    col = eng.create_collection("repo_symbols", dim=64, partition_by="scope_key",
                                embedder=AlgentaBigramEmbedder(), filter_columns=["repo_id", "path", "language"])
    col.add_texts(["class RefundService: ..."], ids=["acme/shop:refund.py:RefundService"],
                  metadata=[{"scope_key": scope_key("acme/shop", "payments", "python"),
                             "repo_id": "acme/shop", "path": "services/payments/refund.py", "language": "python"}])
    hits = col.search_text("refund pending after migration", top_k=20,
                           where={"scope_key": scope_key("acme/shop", "payments", "python")}, explain=True)
    col.save()                       # durable; Telys("./codna-memory").open_collection("repo_symbols") reopens
"""
from __future__ import annotations

import json
import os

import numpy as np

from telys.runtime import FORMAT_VERSION, load_runtime  # SDK↔engine seam (D-30); facade imports no engine

# Single source of truth = the installed distribution version (pyproject). Falls back to a literal only for an
# uninstalled source checkout, so `telys version` / telemetry can never drift from the published wheel.
try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("telys")
except Exception:  # noqa: BLE001 — source tree / not installed
    __version__ = "0.0.0+source"
_SEP = "\x1f"


def scope_key(*parts) -> str:
    """Canonical composite partition key, e.g. scope_key(repo_id, service, language). The collection
    partitions on this single physical key; store the individual fields as filter_columns for off-key
    filtering + explanation."""
    return _SEP.join(str(p) for p in parts)


class Collection:
    """One named, persisted vector collection with an external-id map, embedder, and metadata routing."""

    def __init__(self, engine: "Telys", name: str, dim: int, partition_by, embedder=None,
                 dtype: str = "f32", filter_columns=(), tuner=None, lexical: bool = False) -> None:
        self.engine = engine
        self.name = name
        self.dim = dim
        self.key_name = partition_by[0] if isinstance(partition_by, (list, tuple)) else partition_by
        self.dtype = dtype
        self.col_names = list(filter_columns)
        self.lexical = lexical         # retain per-row tokens at ingest so build_lexical()/mode="lexical" work
        self.embedder = embedder
        self._rt = engine._runtime                 # the SDK↔engine handle (loaded runtime)
        self.tuner = tuner if tuner is not None else self._rt.default_tuner()  # HeuristicTuner (closed runtime)
        self.default_target_recall = 1.0           # raised/lowered by apply_tuning()
        if embedder is not None and embedder.profile.dimension != dim:
            raise ValueError(f"embedder dim {embedder.profile.dimension} != collection dim {dim}")
        self.idx = self._rt.new_index(dim, dtype)  # opaque engine handle; the facade never introspects it
        self._id2int: dict = {}        # external id (str/int) -> internal dense int
        self._int2id: list = []        # internal int -> external id
        self._dir = os.path.join(engine.path, name)

    # ── external <-> internal id mapping (lets callers use "case-101" etc.) ────────────────────────
    def _to_int(self, ext) -> int:
        i = self._id2int.get(ext)
        if i is None:
            i = len(self._int2id); self._id2int[ext] = i; self._int2id.append(ext)
        return i

    def _ext(self, internal_ids):
        return [self._int2id[int(i)] for i in internal_ids]

    def __len__(self):
        return len(self._int2id)

    def __contains__(self, ext):
        return ext in self._id2int

    # ── ingest (explicit add vs upsert; never two physical rows for one logical id) ─────────────────
    def _cols(self, metadata, mask=None):
        md = metadata if mask is None else [m for m, keep in zip(metadata, mask) if keep]
        return {c: np.array([m.get(c) for m in md], dtype=object) for c in self.col_names}

    @staticmethod
    def _mask_texts(texts, mask=None):
        if texts is None:
            return None
        return list(texts) if mask is None else [t for t, keep in zip(texts, mask) if keep]

    def _ingest(self, vecs, ext_ids, metadata, upsert: bool, texts=None):
        vecs = np.ascontiguousarray(vecs, np.float32)
        # Enforce the n*dim contract at the trusted boundary: the runtime strides buffers by the collection
        # dim, so a wrong-width array would cause an out-of-bounds read in the native scan. Fail loudly here.
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"vectors must be (n, {self.dim}); got shape {vecs.shape}")
        keys = np.array([m[self.key_name] for m in metadata])
        if not self._rt.sealed(self.idx):                          # first batch seals the base
            dup = [e for e in ext_ids if e in self._id2int]
            if dup and not upsert:
                raise KeyError(f"ids already exist: {dup[:3]}")
            iids = np.array([self._to_int(e) for e in ext_ids], np.int64)
            if self.embedder is not None:
                self._rt.attach_embedder(self.idx, self.embedder)
            self._rt.build(self.idx, vecs, iids, keys, self.key_name, self._cols(metadata), texts=texts)
            return self._ext(iids)
        new_mask = np.array([e not in self._id2int for e in ext_ids])
        if (~new_mask).any() and not upsert:
            dup = [ext_ids[i] for i in range(len(ext_ids)) if not new_mask[i]]
            raise KeyError(f"ids already exist (use upsert): {dup[:3]}")
        iids = np.array([self._to_int(e) for e in ext_ids], np.int64)
        if new_mask.any():                                         # brand-new rows -> delta insert
            self._rt.insert(self.idx, vecs[new_mask], iids[new_mask], keys[new_mask],
                            self._cols(metadata, new_mask), texts=self._mask_texts(texts, new_mask))
        if (~new_mask).any():                                      # existing ids -> versioned update
            ex = ~new_mask
            self._rt.update(self.idx, iids[ex], vecs[ex], keys[ex], self._cols(metadata, ex),
                            texts=self._mask_texts(texts, ex))
        return self._ext(iids)

    def add_texts(self, texts, ids, metadata):
        """Add NEW rows (raises on any existing id)."""
        if self.embedder is None:
            raise RuntimeError("no embedder attached — create_collection(..., embedder=...) or use add()")
        tv = list(texts)
        return self._ingest(self.embedder.embed_documents(tv), ids, metadata, upsert=False,
                            texts=tv if self.lexical else None)

    def add(self, vectors, ids, metadata):
        return self._ingest(vectors, ids, metadata, upsert=False)

    def upsert_texts(self, texts, ids, metadata):
        """Add new + replace existing (existing ids -> a new visible version, never a duplicate row)."""
        if self.embedder is None:
            raise RuntimeError("no embedder attached")
        tv = list(texts)
        return self._ingest(self.embedder.embed_documents(tv), ids, metadata, upsert=True,
                            texts=tv if self.lexical else None)

    def upsert(self, vectors, ids, metadata):
        return self._ingest(vectors, ids, metadata, upsert=True)

    # ── query (returns external ids) ────────────────────────────────────────────────────────────────
    def search(self, vector, top_k: int = 10, where=None, explain: bool = False,
               target_recall: float | None = None, with_metadata: bool = False):
        # The runtime strides the query by the collection dim; reject a wrong-length query before the native call.
        if np.asarray(vector, np.float32).reshape(-1).shape[0] != self.dim:
            raise ValueError(f"query vector must have length {self.dim}; got {np.asarray(vector).shape}")
        tr = self.default_target_recall if target_recall is None else target_recall
        res = self._rt.search(self.idx, vector, top_k, where, explain, tr)
        if explain:
            ids, sc, ex = res
            out = {"ids": self._ext(ids), "scores": sc.tolist(), "explain": ex}
        else:
            ids, sc = res
            out = {"ids": self._ext(ids), "scores": sc.tolist()}
        if with_metadata:                                  # provenance with the hit (the stored filter columns)
            out["metadata"] = self._rt.columns_for(self.idx, ids)
        return out

    def search_text(self, text, top_k: int = 10, where=None, explain: bool = False,
                    target_recall: float | None = None, with_metadata: bool = False, mode: str = "dense"):
        """mode="dense" (default): embed + filtered vector search. mode="lexical": on-device BM25 over the
        retained code tokens (no embedding), scoped by the same where filter. Requires build_lexical()."""
        if mode == "lexical":
            return self._search_lexical(text, top_k, where=where, explain=explain, with_metadata=with_metadata)
        if self.embedder is None:
            raise RuntimeError("no embedder attached")
        qv = self.embedder.embed_queries([text])[0]
        return self.search(qv, top_k, where=where, explain=explain, target_recall=target_recall,
                           with_metadata=with_metadata)

    def _search_lexical(self, text, top_k, where=None, explain=False, with_metadata=False):
        if not self.lexical:
            raise RuntimeError("collection was not created with lexical=True — mode='lexical' has no index "
                               "(create_collection(..., lexical=True) then build_lexical())")
        res = self._rt.search_lexical(self.idx, text, top_k, where, explain)
        if explain:
            ids, sc, ex = res
            out = {"ids": self._ext(ids), "scores": sc.tolist(), "explain": ex}
        else:
            ids, sc = res
            out = {"ids": self._ext(ids), "scores": sc.tolist()}
        if with_metadata:
            out["metadata"] = self._rt.columns_for(self.idx, ids)
        return out

    def ids(self, where=None) -> list:
        """Live external ids in the collection, optionally scoped (dict/Eq on the partition key or an
        off-key column). Enumerates what a scope currently contains — no consumer-side id map needed."""
        return self._ext(self._rt.live_ids(self.idx, where))

    # ── governed-optional auto-tuning (pluggable Tuner) ───────────────────────────────────────────
    def tune(self, workload=None, objectives=None, dry_run: bool | None = None):
        """Produce a TuningPlan via the collection's Tuner (HeuristicTuner by default; AlgentaTuner when
        configured). The plan is always returned; whether it is APPLIED is decided here (never inside the
        tuner): dry_run=True never applies (an explicit dry_run always wins); dry_run=False always applies;
        dry_run=None (default) applies only if the tuner opts in via mode='suggest_then_apply'."""
        plan = self.tuner.tune_collection(self, workload=workload, objectives=objectives)
        apply = (dry_run is False) or (dry_run is None and getattr(self.tuner, "mode", None) == "suggest_then_apply")
        if apply:
            self.apply_tuning(plan)
        return plan

    def apply_tuning(self, plan):
        """Apply a TuningPlan to the physical layout: default target recall, exact↔quantized crossover, and
        per-partition IVF for oversized partitions (nprobe calibrated to the plan's recall floor)."""
        self.default_target_recall = float(plan.target_recall)
        if plan.exact_crossover_rows and plan.exact_crossover_rows < (1 << 60):
            self._rt.set_exact_crossover(self.idx, int(plan.exact_crossover_rows))
        if plan.ivf.get("enabled"):
            parts = plan.ivf.get("partitions", [])
            built = self._rt.partition_ivf_keys(self.idx)
            # Skip the (expensive) re-clustering when the plan's IVF partitions are already built — e.g. an
            # AlgentaTuner already built them during optimization, or they were restored from disk on reopen.
            need_build = (not parts) or any(p.get("key") not in built for p in parts)
            if need_build:
                self._rt.build_partition_ivf(self.idx, int(plan.ivf["min_rows"]), float(plan.ivf["target_recall"]))
                built = self._rt.partition_ivf_keys(self.idx)
            # A governed tuner (e.g. AlgentaTuner) may carry an optimized nprobe per partition; honor it
            # over the engine's per-partition calibration when present in the plan.
            for p in parts:
                key = p.get("key")
                if p.get("nprobe") and key in built:
                    self._rt.set_partition_nprobe(self.idx, key, int(p["nprobe"]))
        return self.stats()

    # ── mutate / maintain ────────────────────────────────────────────────────────────────────────────
    def delete(self, ids):
        return self._rt.delete(self.idx, [self._id2int[e] for e in ids if e in self._id2int])

    def update_texts(self, texts, ids, metadata):
        """Replace the vector (and optionally key/columns) of EXISTING rows (tombstone-old + append-new).

        Strict by contract (docs/guides/update-records.md): every id must already be LIVE — an unknown or
        deleted id raises instead of silently inserting a stray record. Mixed new+existing batches belong
        in upsert_texts."""
        if self.embedder is None:
            raise RuntimeError("no embedder attached")
        live = {str(e) for e in self.ids()}
        missing = [str(i) for i in ids if str(i) not in live]
        if missing:
            shown = ", ".join(missing[:8]) + (f", … (+{len(missing) - 8} more)" if len(missing) > 8 else "")
            raise ValueError(f"update_texts: ids do not exist (use upsert_texts to insert): {shown}")
        tv = list(texts)
        return self._ingest(self.embedder.embed_documents(tv), ids, metadata, upsert=True,
                            texts=tv if self.lexical else None)

    def compact(self):
        return self._rt.compact(self.idx)

    def build_ivf(self, min_rows: int = 20000, target_recall: float = 0.98):
        return self._rt.build_partition_ivf(self.idx, min_rows, target_recall)

    def build_lexical(self, k1: float = 1.8, b: float = 1.0):
        """Fit the on-device BM25 lexical index over the retained code tokens (per partition). Rebuild after
        inserts/updates/compaction to index new rows (deletes are honored at query time either way). Requires
        the collection was created with lexical=True."""
        if not self.lexical:
            raise RuntimeError("collection was not created with lexical=True (no tokens retained at ingest)")
        return self._rt.build_lexical(self.idx, k1, b)

    def snapshot(self):
        return self._rt.snapshot(self.idx)

    def stats(self):
        s = dict(self._rt.stats(self.idx)); s["name"] = self.name; s["key_name"] = self.key_name
        s["external_ids"] = len(self._int2id); return s

    # ── persistence ──────────────────────────────────────────────────────────────────────────────────
    def save(self):
        self._rt.save(self.idx, self._dir)
        # id_map (external id per row) goes to a BINARY sidecar, not collection.json: at millions of symbols a
        # JSON array of every id is a hundreds-of-MB parse on every open, defeating the mmap'd vector slabs.
        # numpy object array round-trips str/int ids exactly and loads far faster. Atomic (tmp + os.replace).
        import numpy as _np
        imap_tmp = os.path.join(self._dir, "id_map.npy.tmp")
        with open(imap_tmp, "wb") as _f:
            _np.save(_f, _np.array(self._int2id, dtype=object), allow_pickle=True)
        os.replace(imap_tmp, os.path.join(self._dir, "id_map.npy"))
        meta = {"name": self.name, "dim": self.dim, "dtype": self.dtype, "key_name": self.key_name,
                "filter_columns": self.col_names, "id_map_sidecar": "id_map.npy", "telys_version": __version__,
                "default_target_recall": float(self.default_target_recall),  # durable so a tuned collection
                "applied_tuner": getattr(self.tuner, "name", None),          # stays tuned across reopen
                "lexical": bool(self.lexical),                               # retained-tokens -> rebuild plex on open
                "embedder_profile": self.embedder.profile.as_dict() if self.embedder is not None else None}
        tmp = os.path.join(self._dir, "collection.json.tmp")
        with open(tmp, "w") as f:                            # collection.json written last = commit point
            json.dump(meta, f)
        os.replace(tmp, os.path.join(self._dir, "collection.json"))
        return self._dir

    def export_compact(self, path: str, mode: str = "int8") -> str:
        """Write an ULTRA-SMALL READ-ONLY serve artifact: quantized slab + remap + provenance columns —
        no f32 base, no external id map, no lexical tokens. This is the "ship the index" tier: a whole
        collection travels as one small directory that any device can open for search.

        Modes: "int8" (~4x smaller than the f32 save, Mojo i8 scan) and "pq" (~32x scan working set —
        the super-large-repo tier; trains a PQ codebook offline, needs faiss). The collection must have
        been created with dtype="int8" (the engine quantizes at seal time, not at export time).

        Open with `Telys.open_compact(path, embedder)`. Artifact ids are INTERNAL ints (the external
        id map is dropped with the f32 base); provenance stays reachable via with_metadata=True."""
        return self._rt.export_compact(self.idx, path, mode)

    @classmethod
    def _open(cls, engine: "Telys", name: str, embedder=None):
        d = os.path.join(engine.path, name)
        meta = json.load(open(os.path.join(d, "collection.json")))
        self = cls.__new__(cls)
        self.engine = engine; self.name = name; self.dim = meta["dim"]; self.dtype = meta["dtype"]
        self.key_name = meta["key_name"]; self.col_names = meta["filter_columns"]; self._dir = d
        self.lexical = bool(meta.get("lexical", False))
        imap = os.path.join(d, "id_map.npy")
        if "id_map" in meta:                                 # back-compat: older inline-JSON id_map
            self._int2id = list(meta["id_map"])
        elif os.path.exists(imap):
            import numpy as _np
            self._int2id = list(_np.load(imap, allow_pickle=True))
        else:
            self._int2id = []
        self._id2int = {e: i for i, e in enumerate(self._int2id)}
        self._rt = engine._runtime
        self.idx = self._rt.open_index(d)
        # HeuristicTuner is the safe re-tune default (an AlgentaTuner can't be restored without its SDK+config);
        # but the APPLIED target recall is restored so a persisted IVF stays reachable on the default query path.
        self.tuner = self._rt.default_tuner()
        self.default_target_recall = float(meta.get("default_target_recall", 1.0))
        self.embedder = None
        prof = meta.get("embedder_profile")
        if embedder is None and prof is not None:           # auto-attach from the engine's provider registry
            embedder = engine._providers.get(prof["model_id"]) or engine._providers.get(prof.get("space_id"))
        if embedder is not None:
            if prof is not None:
                from telys.embedding import EmbeddingProfile
                stored = EmbeddingProfile(**{k: v for k, v in prof.items() if k != "space_id"})
                if not embedder.profile.compatible_with(stored):
                    raise ValueError(f"provider space {embedder.profile.space_id()} != collection "
                                     f"space {stored.space_id()} (model/version/dim/normalization/metric/tokenizer)")
            self._rt.attach_embedder(self.idx, embedder); self.embedder = embedder
        if self.lexical:                                     # rebuild the per-partition BM25 from persisted tokens
            try:
                self._rt.build_lexical(self.idx)
            except NotImplementedError:                       # native runtime: dense still opens, lexical inert
                pass
        return self


class CompactCollection:
    """Read-only search surface over a compact serve artifact (`Collection.export_compact`).

    The artifact keeps the quantized slab, the partition directory, and the provenance columns —
    and drops the f32 base, the external id map, and the lexical tokens. Consequences, all by
    design: ids come back as INTERNAL ints (resolve provenance with with_metadata=True); mutation
    has no surface here; off-key (non-partition) filters are refused by the engine. Vector search
    only — the Mojo quantized scan path."""

    def __init__(self, engine: "Telys", idx, embedder=None) -> None:
        self.engine = engine
        self._rt = engine._runtime
        self.idx = idx
        self.embedder = embedder
        if embedder is not None:
            self._rt.attach_embedder(idx, embedder)

    def search(self, vector, top_k: int = 10, where=None, explain: bool = False,
               target_recall: float | None = None, with_metadata: bool = False):
        tr = 1.0 if target_recall is None else float(target_recall)
        res = self._rt.search(self.idx, vector, top_k, where, explain, tr)
        if explain:
            ids, sc, ex = res
            out = {"ids": [int(i) for i in ids], "scores": sc.tolist(), "explain": ex}
        else:
            ids, sc = res
            out = {"ids": [int(i) for i in ids], "scores": sc.tolist()}
        if with_metadata:
            out["metadata"] = self._rt.columns_for(self.idx, ids)
        return out

    def search_text(self, text, top_k: int = 10, where=None, explain: bool = False,
                    target_recall: float | None = None, with_metadata: bool = False):
        if self.embedder is None:
            raise RuntimeError("no embedder attached — open_compact(path, embedder=...)")
        qv = self.embedder.embed_queries([text])[0]
        return self.search(qv, top_k, where=where, explain=explain, target_recall=target_recall,
                           with_metadata=with_metadata)

    def stats(self) -> dict:
        s = dict(self._rt.stats(self.idx))
        s["compact"] = True
        return s


class Telys:
    """An embedded engine rooted at a directory; manages named collections. (Alias: AME.)"""

    def __init__(self, path: str, embedding_providers: dict | None = None, runtime=None) -> None:
        self.path = path
        os.makedirs(path, exist_ok=True)
        self._cols: dict = {}
        # The SDK↔engine handle, shared by this engine's collections. Loaded lazily here so `import telys`
        # never imports the engine; a native runtime can be injected via `runtime=` (D-30).
        self._runtime = runtime if runtime is not None else load_runtime()
        # registry: model_id (or space_id) -> EmbeddingProvider, used to auto-attach on open_collection
        self._providers: dict = dict(embedding_providers or {})

    def register_provider(self, key: str, provider) -> None:
        self._providers[key] = provider

    @staticmethod
    def scope_key(*parts) -> str:
        """Handle form of the module-level `telys.scope_key` (docs/understand/partitions-scope-key.md)."""
        return scope_key(*parts)

    def create_collection(self, name: str, dim: int, partition_by, embedder=None,
                          dtype: str = "f32", filter_columns=(), tuner=None, lexical: bool = False) -> Collection:
        c = Collection(self, name, dim, partition_by, embedder, dtype, filter_columns, tuner, lexical)
        self._cols[name] = c
        return c

    def open_collection(self, name: str, embedder=None) -> Collection:
        c = Collection._open(self, name, embedder)
        self._cols[name] = c
        return c

    def open_compact(self, path: str, embedder=None) -> CompactCollection:
        """Open a compact serve artifact written by `Collection.export_compact` — read-only search
        over the quantized slab. The artifact is self-contained (no engine directory needed)."""
        idx = self._runtime.open_compact(path)
        return CompactCollection(self, idx, embedder)

    def collections(self) -> list:
        return sorted(n for n in os.listdir(self.path)
                      if os.path.exists(os.path.join(self.path, n, "collection.json")))

    def __getitem__(self, name: str) -> Collection:
        return self._cols.get(name) or self.open_collection(name)


AME = Telys  # backward-compatible alias (the engine was prototyped as AME)
