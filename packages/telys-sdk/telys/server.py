"""Telys self-host server — the Team deployment (D-30 server tier).

A team shares ONE on-device memory by self-hosting this daemon on their own box/VPC: it loads the engine once
(in-process) and serves the Collection facade over a Unix-domain socket. Every connected client shares the
SAME engine — so a row added by one teammate is immediately visible to another — with `partition_by` giving
either shared (one partition) or per-member-isolated (per-tenant partition) memory. No third-party cloud: the
data never leaves the team's machine.

Why this lives server-side: the external↔internal id map and the embedder live in the engine's `Collection`,
so for a *shared* memory they must be shared — i.e. owned by the server, not duplicated per client. Clients
therefore speak the high-level facade ops (create/add/search/…), not the low-level index handle.

Concurrency: the engine's own writer-preferring RWLock guards the native structures; this server adds a
per-collection RWLock around the facade ops (the Python id-map) so reads run in parallel while a write is
exclusive — preserving the engine's parallel-reader property end to end.

License: optionally verifies a `telys` (ver:2) entitlement offline on boot (the Team tier), refusing to serve
without it. The data path stays fully offline thereafter.

    telys serve --path ./team-memory --socket /tmp/telys.sock
    # client:  from telys.client import connect;  eng = connect("/tmp/telys.sock")
"""
from __future__ import annotations

import logging
import os
import socket
import threading

import numpy as np

from telys._wire import recv_msg, send_msg

_log = logging.getLogger("telys.server")


class _RWLock:
    """Minimal readers-writer lock: parallel readers, exclusive writer (writer-preferring to avoid starvation)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writers_waiting = 0
        self._writer = False

    def acquire_read(self) -> None:
        with self._cond:
            while self._writer or self._writers_waiting:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._writers_waiting += 1
            while self._writer or self._readers:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writer = True

    def release_write(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()


class TelysServer:
    """Serves one in-process Telys engine to many sharing clients over a Unix socket OR a TCP port."""

    # ops that mutate the collection (take the per-collection write lock); everything else takes a read lock
    _WRITE_OPS = frozenset({"add", "upsert", "add_texts", "upsert_texts", "delete", "compact", "save",
                            "build_ivf", "snapshot"})

    _AUTH_TIMEOUT_S = 30.0   # an unauthenticated connection must complete the handshake within this window

    def __init__(self, path: str, *, embedder_factory=None, license_token: str | None = None,
                 require_license: bool = False, access_token: str | None = None,
                 save_interval_s: float = 0.0) -> None:
        # Optional Team-tier license gate (offline). Verified BEFORE the engine serves any data.
        if require_license:
            self._verify_license(license_token)
        # An empty/blank token would silently disable auth — refuse it (None = intentionally no auth).
        if access_token is not None and (not isinstance(access_token, str) or not access_token.strip()):
            raise ValueError("access_token must be a non-empty string (or None for a trusted Unix socket)")
        # Import the engine lazily (keeps `import telys` engine-free; the server is a runtime-side artifact).
        from telys.engine import Telys
        self._engine = Telys(path)
        self._embedder_factory = embedder_factory       # spec -> EmbeddingProvider (overrides the built-ins)
        self._access_token = access_token               # shared bearer for TCP (None on a trusted Unix socket)
        self._save_interval_s = float(save_interval_s)  # >0 → background periodic save (bounds crash data loss)
        self._locks: dict[str, _RWLock] = {}            # per-collection RWLock
        self._reg_lock = threading.Lock()               # guards _locks + collection creation/open
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._register_builtin_embedders()

    # ── durability: periodic + final save (bounds data loss if the in-process engine dies) ───────────
    def _save_all_open(self) -> int:
        """Persist every open collection under its write lock. Best-effort: a failure on one collection is
        logged and does not abort the others (a crash must not be made worse by a save loop)."""
        saved = 0
        with self._reg_lock:
            names = list(self._engine._cols.keys())
        for name in names:
            try:
                lock = self._lock_for(name)
                lock.acquire_write()
                try:
                    self._engine._cols[name].save()
                    saved += 1
                finally:
                    lock.release_write()
            except Exception as exc:  # noqa: BLE001
                _log.warning("periodic save failed for collection %r: %s", name, exc)
        return saved

    def _lock_for(self, name: str) -> "_RWLock":
        with self._reg_lock:
            return self._locks.setdefault(name, _RWLock())

    def _periodic_save_loop(self) -> None:
        while not self._stop.wait(self._save_interval_s):   # wait() returns True when stopped → exits promptly
            n = self._save_all_open()
            if n:
                _log.debug("periodic save: %d collection(s)", n)

    def _token_ok(self, tok) -> bool:
        import hmac
        return (self._access_token is not None and isinstance(tok, str)
                and hmac.compare_digest(tok, self._access_token))

    # ── license (Team tier) ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _verify_license(token: str | None) -> None:
        if not token:
            raise PermissionError(
                "telys serve requires a Telys license (Team tier). Pass --license <token> "
                "(or TELYS_LICENSE), or run without --require-license for local/dev."
            )
        from telys import verify as V
        import time
        V.verify_license(token, now=int(time.time()))  # raises VerificationError unless products.telys present

    # ── collection registry (server-owned, shared by all clients) ──────────────────────────────────
    @staticmethod
    def _check_name(name) -> str:
        # A client-supplied name flows into the filesystem (engine.path/<name>). Reject anything that isn't a
        # single safe path component — no traversal ('..'), separators, or NULs (path-traversal hardening).
        if (not isinstance(name, str) or name in ("", ".", "..")
                or os.path.basename(name) != name
                or any(c in name for c in ("/", "\\", "\x00"))):
            raise ValueError(f"invalid collection name: {name!r}")
        return name

    def _get_col(self, name: str):
        # MUST be called under self._reg_lock: a lock-free read-then-open races two first-touching clients into
        # two divergent Collection objects (independent id maps) — which would silently UN-share the memory.
        col = self._engine._cols.get(name)
        if col is None:                                  # opened lazily from disk if it exists
            col = self._engine.open_collection(name)
        return col

    def _resolve(self, name: str):
        """Atomically resolve (collection, per-collection lock) under the registry lock."""
        with self._reg_lock:
            return self._get_col(name), self._locks.setdefault(name, _RWLock())

    # ── op dispatch ────────────────────────────────────────────────────────────────────────────────
    def _dispatch(self, req: dict):
        op = req.get("op")
        name = req.get("collection")
        a = req.get("args") or {}

        if op == "ping":
            return {"pong": True}
        if op == "collections":
            return self._engine.collections()

        self._check_name(name)  # every remaining op is collection-scoped

        if op == "create_collection":
            want_key = a["partition_by"][0] if isinstance(a["partition_by"], (list, tuple)) else a["partition_by"]
            want_cols = tuple(a.get("filter_columns", ()))
            with self._reg_lock:
                exists = name in self._engine._cols or os.path.exists(
                    os.path.join(self._engine.path, name, "collection.json"))
                if exists:
                    # idempotent for a SECOND sharing client — but the schema must match, else its later
                    # add()s would fail confusingly against the existing layout.
                    col = self._get_col(name)
                    if (col.dim != int(a["dim"]) or col.key_name != want_key
                            or tuple(col.col_names) != want_cols):
                        raise ValueError(
                            f"collection {name!r} already exists with a different schema "
                            f"(dim={col.dim}, partition_by={col.key_name!r}, "
                            f"filter_columns={tuple(col.col_names)})")
                else:
                    self._engine.create_collection(
                        name, int(a["dim"]), a["partition_by"],
                        embedder=self._make_embedder(a.get("embedder")), dtype=a.get("dtype", "f32"),
                        filter_columns=want_cols)
                self._locks.setdefault(name, _RWLock())
            return {"created": name}

        if op == "open_collection":
            self._resolve(name)
            return {"opened": name}

        if op == "drop_collection":
            # Remove a collection from the shared engine + delete its on-disk dir (name is path-checked above).
            # Needed for benchmark harnesses that recreate a fixed collection between runs (drop_old).
            import shutil
            with self._reg_lock:
                self._engine._cols.pop(name, None)
                self._locks.pop(name, None)
                d = os.path.join(self._engine.path, name)
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
            return {"dropped": name}

        # data-path ops: resolve collection + lock atomically, then take the per-collection RWLock
        col, lock = self._resolve(name)
        write = op in self._WRITE_OPS
        lock.acquire_write() if write else lock.acquire_read()
        try:
            return self._run_op(op, col, a)
        finally:
            lock.release_write() if write else lock.release_read()

    def _run_op(self, op: str, col, a: dict):
        if op == "add":
            return col.add(self._vecs(a["vectors"]), a["ids"], a["metadata"])
        if op == "upsert":
            return col.upsert(self._vecs(a["vectors"]), a["ids"], a["metadata"])
        if op == "add_texts":
            return col.add_texts(a["texts"], a["ids"], a["metadata"])
        if op == "upsert_texts":
            return col.upsert_texts(a["texts"], a["ids"], a["metadata"])
        if op == "search":
            return col.search(self._vec(a["vector"]), top_k=self._topk(a),
                              where=a.get("where"), explain=bool(a.get("explain", False)),
                              with_metadata=bool(a.get("with_metadata", False)))
        if op == "search_text":
            return col.search_text(a["text"], top_k=self._topk(a),
                                  where=a.get("where"), explain=bool(a.get("explain", False)),
                                  with_metadata=bool(a.get("with_metadata", False)))
        if op == "ids":
            return col.ids(where=a.get("where"))
        if op == "delete":
            return col.delete(a["ids"])
        if op == "compact":
            return col.compact()
        if op == "save":
            return col.save()
        if op == "stats":
            return col.stats()
        if op == "build_ivf":
            return col.build_ivf(min_rows=int(a.get("min_rows", 20000)),
                                 target_recall=float(a.get("target_recall", 0.98)))
        if op == "snapshot":
            return col.snapshot()
        raise ValueError(f"unknown op: {op!r}")

    # ── helpers ────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _vecs(v):
        return np.ascontiguousarray(np.asarray(v, dtype=np.float32))

    @staticmethod
    def _vec(v):
        return np.ascontiguousarray(np.asarray(v, dtype=np.float32).reshape(-1))

    @staticmethod
    def _topk(a) -> int:
        return max(1, min(int(a.get("top_k", 10)), 100_000))  # bound a misbehaving client

    def _register_builtin_embedders(self) -> None:
        """Pre-register the bundled on-device embedder(s) so add_texts/search_text work AND a reopened text
        collection auto-reattaches its embedder (Collection._open resolves from engine._providers by model_id).
        Best-effort: the bigram embedder calls the native kernel, so if the runtime kernel is unavailable the
        server still serves vector-only collections (add()/search()) — text ops then return a clear error."""
        try:
            from telys.embedding import AlgentaBigramEmbedder
            emb = AlgentaBigramEmbedder()                       # 64-d, on-device, zero network
            self._engine.register_provider(emb.profile.model_id, emb)   # "bigram" (used by _open auto-attach)
            self._engine.register_provider("bigram", emb)               # friendly alias for create_collection
        except Exception:  # noqa: BLE001 — kernel/runtime not available; vector-only server
            pass

    def _make_embedder(self, spec):
        """Resolve an embedder spec to a provider. A custom embedder_factory (if given) wins; otherwise the
        spec names a pre-registered built-in (e.g. 'bigram'). The embedder lives server-side so every sharing
        client embeds into the SAME space — the point of a shared memory."""
        if not spec:
            return None
        if self._embedder_factory is not None:
            return self._embedder_factory(spec)
        name = spec.get("name") if isinstance(spec, dict) else spec
        emb = self._engine._providers.get(name)
        if emb is None:
            raise ValueError(
                f"unknown embedder {name!r}: this server provides 'bigram' (requires the runtime kernel). "
                "Pass embedder_factory= for a custom provider, or use add()/search() with vectors.")
        return emb

    # ── connection + accept loop ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _errstr(exc) -> str:
        try:
            return f"{type(exc).__name__}: {exc}"
        except Exception:  # noqa: BLE001 — a pathological __str__ must not crash the error path
            return type(exc).__name__

    @staticmethod
    def _safe_send(conn: socket.socket, obj) -> bool:
        try:
            send_msg(conn, obj)
            return True
        except (ConnectionError, OSError):
            return False

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            authed = self._access_token is None       # no token configured -> trusted transport (Unix socket)
            if not authed:
                conn.settimeout(self._AUTH_TIMEOUT_S)  # bound a slow-loris that never authenticates
            while not self._stop.is_set():
                try:
                    req = recv_msg(conn)
                except socket.timeout:
                    return                                   # pre-auth client too slow -> drop the connection
                except (ConnectionError, OSError):
                    return                                   # clean disconnect
                except Exception:  # noqa: BLE001 — malformed/oversized frame: stream may be desynced
                    self._safe_send(conn, {"ok": False, "error": "bad request"})
                    return                                   # close; the client can reconnect
                op = req.get("op")
                # Auth handshake: when an access token is configured, the FIRST message must be a valid `auth`.
                if op == "auth":
                    if authed:
                        self._safe_send(conn, {"ok": True, "result": {"authenticated": True}})
                        continue                             # already authed: a later auth is a no-op, never re-checked
                    if self._token_ok((req.get("args") or {}).get("token")):
                        authed = True
                        conn.settimeout(None)                # authed -> persistent connection, no idle timeout
                        self._safe_send(conn, {"ok": True, "result": {"authenticated": True}})
                        continue
                    self._safe_send(conn, {"ok": False, "error": "unauthorized"})
                    return                                   # wrong token -> close
                if not authed:
                    self._safe_send(conn, {"ok": False, "error": "unauthorized: send auth first"})
                    return
                # dispatch and reply are ISOLATED: a broken pipe on the reply must never re-enter the error
                # path (which would try to send again and kill the handler thread).
                try:
                    reply = {"ok": True, "result": self._dispatch(req)}
                except Exception as exc:  # noqa: BLE001 — surface to the client, keep serving
                    _log.warning("op %r failed: %s: %s", req.get("op"), type(exc).__name__, exc)
                    reply = {"ok": False, "error": self._errstr(exc)}
                if not self._safe_send(conn, reply):
                    return

    def serve_forever(self, socket_path: str | None = None, *, host: str | None = None,
                      port: int | None = None, ready: threading.Event | None = None) -> None:
        """Listen on a Unix socket (``socket_path``) or a TCP endpoint (``host``+``port``).

        Unix: owner-only (0o600), the socket file is the trust boundary on a shared host — no token needed.
        TCP: for a containerised / cross-machine team deployment; pair with access_token= and run it on a
        trusted/private network (TLS is terminated by a proxy — out of MVP scope)."""
        is_unix = socket_path is not None
        if is_unix:
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.bind(socket_path)
            os.chmod(socket_path, 0o600)
        elif host is not None and port is not None:
            if self._access_token is None:
                # A network-exposed server with no auth is a footgun — refuse rather than serve open.
                raise ValueError("TCP serving requires access_token= (no anonymous network access)")
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((host, int(port)))
        else:
            raise ValueError("serve_forever needs either socket_path or host+port")
        self._sock.listen(64)
        self._sock.settimeout(0.5)
        self._install_signal_handlers()                 # SIGTERM/SIGINT → graceful drain (main thread only)
        if self._save_interval_s > 0:                   # background periodic save bounds crash data loss
            threading.Thread(target=self._periodic_save_loop, daemon=True).start()
        if ready is not None:
            ready.set()
        try:
            while not self._stop.is_set():
                try:
                    conn, _ = self._sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
        finally:
            self._stop.set()                            # stop the periodic-save loop too
            self._sock.close()
            if is_unix and os.path.exists(socket_path):
                os.unlink(socket_path)
            try:
                self._save_all_open()                   # graceful drain: persist everything on the way out
            except Exception as exc:                    # noqa: BLE001
                _log.warning("final save failed: %s", exc)

    def _install_signal_handlers(self) -> None:
        # signal.signal() only works on the main thread; tests run serve_forever in a worker thread, so this
        # no-ops there. The CLI runs it on the main thread → SIGTERM/SIGINT trigger a graceful drain + save.
        import signal
        if threading.current_thread() is not threading.main_thread():
            return

        def _drain(signum, _frame):
            _log.info("received signal %s — draining + saving", signum)
            self._stop.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _drain)
            except (ValueError, OSError):               # not main thread / unsupported platform
                pass

    def stop(self) -> None:
        self._stop.set()


def serve(path: str, socket_path: str | None = None, *, host: str | None = None, port: int | None = None,
          license_token: str | None = None, require_license: bool = False,
          access_token: str | None = None, embedder_factory=None, save_interval_s: float = 0.0) -> TelysServer:
    """Run a Telys self-host server (blocking). Unix socket or TCP (host+port, requires access_token)."""
    srv = TelysServer(path, embedder_factory=embedder_factory, license_token=license_token,
                      require_license=require_license, access_token=access_token, save_interval_s=save_interval_s)
    srv.serve_forever(socket_path, host=host, port=port)
    return srv
