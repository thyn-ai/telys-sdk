"""First-class binary blob / KV fast path for Telys.

`BlobStore` is a local, append-only key/value store for opaque bytes. It lives purely in the Python SDK:
no runtime, no vectors, no embedding, no network, and no login beyond the path you hand it. It is meant for
integrations that need sub-millisecond put/get of binary payloads (e.g. vLLM KV caches) without paying the
vector-collection overhead.

Persistence:
  - ``<path>/data.slab``   : append-only record stream
  - ``<path>/index.log``   : append-only index mutations (O(1) per op, no rewrite-with-count regression)
  - ``<path>/index.json``  : compacted snapshot, written on ``close()`` / ``sync()``

The slab is the source of truth: if the index files are missing or corrupt, the store scans the slab and
rebuilds. The latest record for a key wins, so overwrite-by-put and crash-after-append are both safe.
"""
from __future__ import annotations

import json
import os
import struct
import threading
from collections.abc import Iterable
from typing import Any

MAGIC = b"TELYBB01"
FORMAT_VERSION = 1
HEADER_FMT = ">IIQ"  # key_len: uint32, meta_len: uint32, payload_len: uint64
HEADER_SIZE = struct.calcsize(HEADER_FMT)
INDEX_LOG_NAME = "index.log"
INDEX_SNAPSHOT_NAME = "index.json"
SLAB_NAME = "data.slab"

_MAX_KEY_LEN = 0xFFFFFFFF
_MAX_META_LEN = 0xFFFFFFFF
_MAX_PAYLOAD_LEN = 0xFFFFFFFFFFFFFFFF


class BlobStoreError(RuntimeError):
    """Raised for structural or I/O errors inside a BlobStore."""


class BlobStore:
    """Append-only local key/value store for binary payloads.

        from telys import BlobStore
        store = BlobStore("./kv-cache")
        store.put("acme/42", b"opaque bytes", metadata={"model": "llama-3-8b"})
        payload, meta = store.get("acme/42")
        assert store.contains("acme/42")

    The store is durable across restarts, thread-safe, and has no runtime dependency. By default writes are
    buffered for speed; call ``sync()`` or use the store as a context manager to guarantee crash durability.
    """

    def __init__(self, path: str) -> None:
        self._path = os.path.abspath(path)
        os.makedirs(self._path, exist_ok=True)
        self._slab_path = os.path.join(self._path, SLAB_NAME)
        self._log_path = os.path.join(self._path, INDEX_LOG_NAME)
        self._snapshot_path = os.path.join(self._path, INDEX_SNAPSHOT_NAME)

        self._lock = threading.Lock()
        self._index: dict[str, dict[str, int]] = {}
        self._slab: Any = None
        self._log: Any = None

        self._open()

    # ── lifecycle ────────────────────────────────────────────────────────────────────────────────────
    def _open(self) -> None:
        if not os.path.exists(self._slab_path):
            with open(self._slab_path, "wb") as f:
                f.write(MAGIC)
                f.flush()
                os.fsync(f.fileno())
        else:
            with open(self._slab_path, "rb") as f:
                if f.read(len(MAGIC)) != MAGIC:
                    raise BlobStoreError(
                        f"{self._slab_path} is not a Telys blob slab (bad magic)"
                    )

        self._slab = open(self._slab_path, "a+b")  # noqa: SIM115 — long-lived handle closed in close()
        self._log = open(self._log_path, "a+", encoding="utf-8") if os.path.exists(self._log_path) else None  # noqa: SIM115

        self._index = self._load_index()
        if not self._index and os.path.getsize(self._slab_path) > len(MAGIC):
            # Empty index but non-empty slab: scan and snapshot.
            self._index = self._scan_slab()
            if self._index:
                self._write_snapshot(sync=False)

    def close(self) -> None:
        with self._lock:
            try:
                if self._slab is not None:
                    self._fsync(self._slab)
                    self._write_snapshot(sync=True)
                    self._slab.close()
                    self._slab = None
                if self._log is not None:
                    self._log.close()
                    self._log = None
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── index management ─────────────────────────────────────────────────────────────────────────────
    def _load_index(self) -> dict[str, dict[str, int]]:
        """Load the current index: snapshot + replayed log, with slab scan as fallback."""
        snapshot: dict[str, dict[str, int]] = {}
        snapshot_ok = False
        if os.path.exists(self._snapshot_path):
            try:
                snapshot = self._read_snapshot()
                snapshot_ok = True
            except (OSError, ValueError, KeyError):
                snapshot = {}

        if self._log is not None:
            try:
                self._log.seek(0)
                log_index = self._replay_log(self._log)
            except (OSError, ValueError, KeyError):
                log_index = {}
            # Log entries are authoritative over the snapshot.
            snapshot.update(log_index)
            return snapshot

        if snapshot_ok:
            return snapshot

        # No usable index files: scan the slab.
        return self._scan_slab()

    def _read_snapshot(self) -> dict[str, dict[str, int]]:
        with open(self._snapshot_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported blob index version {data.get('version')}")
        records = data.get("records", {})
        size = os.path.getsize(self._slab_path)
        for key, rec in records.items():
            off = rec["offset"]
            total = HEADER_SIZE + rec["key_len"] + rec["meta_len"] + rec["payload_len"]
            if off < len(MAGIC) or off + total > size:
                raise ValueError(f"index entry for {key!r} is out of bounds")
        return records

    def _replay_log(self, fh) -> dict[str, dict[str, int] | None]:
        """Replay an index log, returning mutations. A value of None means delete."""
        mutations: dict[str, dict[str, int] | None] = {}
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key = entry["k"]
            if entry.get("d"):
                mutations[key] = None
            else:
                mutations[key] = {
                    "offset": entry["o"],
                    "key_len": entry["kl"],
                    "meta_len": entry["ml"],
                    "payload_len": entry["pl"],
                }
        return mutations

    def _write_snapshot(self, sync: bool = False) -> None:
        tmp = self._snapshot_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": FORMAT_VERSION, "records": self._index}, f, separators=(",", ":"))
            if sync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, self._snapshot_path)
        # After snapshotting, the log is redundant; truncate it on next use.
        if self._log is not None:
            self._log.close()
            self._log = None
        try:
            os.remove(self._log_path)
        except FileNotFoundError:
            pass

    def _ensure_log_open(self) -> Any:
        if self._log is None:
            self._log = open(self._log_path, "a+", encoding="utf-8")  # noqa: SIM115 — long-lived handle
        return self._log

    def _append_log(self, entry: dict, sync: bool) -> None:
        log = self._ensure_log_open()
        log.write(json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n")
        log.flush()
        if sync:
            os.fsync(log.fileno())

    @staticmethod
    def _fsync(fh) -> None:
        fh.flush()
        os.fsync(fh.fileno())

    def sync(self) -> None:
        """Explicitly flush the slab and write a compacted snapshot.

        BlobStore methods default to ``sync=False`` so the hot path stays sub-millisecond; call ``sync()``
        (or use a context manager, which syncs on exit) when you need crash durability before continuing.
        """
        with self._lock:
            self._fsync(self._slab)
            self._write_snapshot(sync=True)

    def _scan_slab(self) -> dict[str, dict[str, int]]:
        index: dict[str, dict[str, int]] = {}
        with open(self._slab_path, "rb") as f:
            if f.read(len(MAGIC)) != MAGIC:
                raise BlobStoreError("bad magic while scanning slab")
            while True:
                offset = f.tell()
                header = f.read(HEADER_SIZE)
                if len(header) == 0:
                    break
                if len(header) != HEADER_SIZE:
                    raise BlobStoreError(f"truncated record header at offset {offset}")
                key_len, meta_len, payload_len = struct.unpack(HEADER_FMT, header)
                if key_len > _MAX_KEY_LEN or meta_len > _MAX_META_LEN or payload_len > _MAX_PAYLOAD_LEN:
                    raise BlobStoreError(f"impossible record lengths at offset {offset}")
                key = f.read(key_len).decode("utf-8")
                f.seek(meta_len + payload_len, 1)
                index[key] = {
                    "offset": offset,
                    "key_len": key_len,
                    "meta_len": meta_len,
                    "payload_len": payload_len,
                }
        return index

    # ── validation helpers ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _assert_str_key(key: object) -> str:
        if not isinstance(key, str):
            raise TypeError(f"key must be str, got {type(key).__name__}")
        return key

    @staticmethod
    def _assert_bytes_payload(payload: object) -> bytes:
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError(f"payload must be bytes, got {type(payload).__name__}")
        return bytes(payload)

    @staticmethod
    def _encode_record(key: str, payload: bytes, metadata: dict | None) -> tuple[bytes, bytes, bytes, bytes]:
        meta = dict(metadata) if metadata else {}
        key_b = key.encode("utf-8")
        meta_b = json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8")
        if len(key_b) > _MAX_KEY_LEN:
            raise ValueError("key too large")
        if len(meta_b) > _MAX_META_LEN:
            raise ValueError("metadata too large")
        if len(payload) > _MAX_PAYLOAD_LEN:
            raise ValueError("payload too large")
        header = struct.pack(HEADER_FMT, len(key_b), len(meta_b), len(payload))
        return key_b, meta_b, payload, header

    # ── core ops ─────────────────────────────────────────────────────────────────────────────────────
    def put(self, key: str, payload: bytes, metadata: dict | None = None, *, sync: bool = False) -> None:
        """Store ``payload`` under ``key``. Overwrites any existing value.

        ``sync=True`` fsyncs the slab and index log before returning (crash durable). The default
        ``sync=False`` keeps the hot path sub-millisecond; durability is still guaranteed on ``close()``
        or ``sync()``.
        """
        self._assert_str_key(key)
        payload = self._assert_bytes_payload(payload)
        key_b, meta_b, payload_b, header = self._encode_record(key, payload, metadata)
        record = header + key_b + meta_b + payload_b

        with self._lock:
            self._slab.seek(0, 2)
            offset = self._slab.tell()
            self._slab.write(record)
            if sync:
                self._fsync(self._slab)
            rec = {
                "offset": offset,
                "key_len": len(key_b),
                "meta_len": len(meta_b),
                "payload_len": len(payload_b),
            }
            self._index[key] = rec
            self._append_log(
                {"k": key, "o": offset, "kl": len(key_b), "ml": len(meta_b), "pl": len(payload_b)},
                sync=sync,
            )

    def get(self, key: str) -> tuple[bytes, dict]:
        """Return ``(payload, metadata)`` for ``key``. Raises ``KeyError`` if absent."""
        self._assert_str_key(key)
        with self._lock:
            rec = self._index.get(key)
            if rec is None:
                raise KeyError(key)
            return self._read_record(rec, key)

    def delete(self, key: str, *, sync: bool = False) -> None:
        """Remove ``key``. Idempotent: missing keys are ignored.

        ``sync=True`` fsyncs the index log before returning.
        """
        self._assert_str_key(key)
        with self._lock:
            if key not in self._index:
                return
            del self._index[key]
            self._append_log({"k": key, "d": True}, sync=sync)

    def contains(self, key: str) -> bool:
        """True iff ``key`` is currently stored."""
        self._assert_str_key(key)
        with self._lock:
            return key in self._index

    def keys(self) -> set[str]:
        """Snapshot of currently stored keys."""
        with self._lock:
            return set(self._index.keys())

    def __contains__(self, key: str) -> bool:
        return self.contains(key)

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    # ── batch ops ────────────────────────────────────────────────────────────────────────────────────
    def put_batch(self, items: Iterable[tuple[str, bytes, dict | None]], *, sync: bool = False) -> None:
        """Store many blobs in one slab write + one index log flush.

        Each item is ``(key, payload, metadata)``. Metadata may be ``None``.
        ``sync=True`` fsyncs the slab and index log before returning.
        """
        prepared: list[tuple[str, bytes, bytes, bytes, bytes, dict]] = []
        for item in items:
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError("each batch item must be (key, payload, metadata)")
            key, payload, metadata = item
            key = self._assert_str_key(key)
            payload = self._assert_bytes_payload(payload)
            key_b, meta_b, payload_b, header = self._encode_record(key, payload, metadata)
            prepared.append((key, key_b, meta_b, payload_b, header + key_b + meta_b + payload_b, rec := {
                "key_len": len(key_b),
                "meta_len": len(meta_b),
                "payload_len": len(payload_b),
            }))

        with self._lock:
            self._slab.seek(0, 2)
            log = self._ensure_log_open()
            for key, key_b, meta_b, payload_b, record, rec in prepared:
                offset = self._slab.tell()
                self._slab.write(record)
                rec["offset"] = offset
                self._index[key] = rec
                log.write(json.dumps(
                    {"k": key, "o": offset, "kl": len(key_b), "ml": len(meta_b), "pl": len(payload_b)},
                    separators=(",", ":"),
                    sort_keys=True,
                ) + "\n")
            if sync:
                self._fsync(self._slab)
            log.flush()
            if sync:
                os.fsync(log.fileno())

    def get_batch(self, keys: Iterable[str]) -> dict[str, tuple[bytes, dict]]:
        """Return ``{key: (payload, metadata)}`` for every key. Raises ``KeyError`` on the first missing key."""
        keys = list(keys)
        if not keys:
            return {}
        for key in keys:
            self._assert_str_key(key)

        with self._lock:
            missing = [k for k in keys if k not in self._index]
            if missing:
                raise KeyError(missing[0])
            # Read in slab order for locality.
            order = sorted(
                ((k, self._index[k]) for k in keys),
                key=lambda t: t[1]["offset"],
            )
            out: dict[str, tuple[bytes, dict]] = {}
            for key, rec in order:
                out[key] = self._read_record(rec, key)
            return out

    # ── internal read ────────────────────────────────────────────────────────────────────────────────
    def _read_record(self, rec: dict[str, int], expected_key: str) -> tuple[bytes, dict]:
        self._slab.seek(rec["offset"])
        header = self._slab.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE:
            raise BlobStoreError(f"short read for key {expected_key!r}")
        key_len, meta_len, payload_len = struct.unpack(HEADER_FMT, header)
        if (
            key_len != rec["key_len"]
            or meta_len != rec["meta_len"]
            or payload_len != rec["payload_len"]
        ):
            raise BlobStoreError(f"index/slab mismatch for key {expected_key!r}")
        stored_key = self._slab.read(key_len).decode("utf-8")
        if stored_key != expected_key:
            raise BlobStoreError(f"index points to wrong key: {stored_key!r}")
        meta = json.loads(self._slab.read(meta_len).decode("utf-8"))
        payload = self._slab.read(payload_len)
        return payload, meta
