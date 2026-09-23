"""Telys self-host client — the thin remote half of the Team deployment.

``connect()`` returns a ``RemoteTelys`` whose facade mirrors the in-process ``Telys``/``Collection`` API, so
application code is unchanged: it just talks to a shared self-hosted engine over a Unix socket instead of an
in-process dylib. Every client of the same server shares one memory.

    from telys.client import connect
    eng = connect("/tmp/telys.sock")
    col = eng.create_collection("mem", dim=768, partition_by="tenant_id", filter_columns=["tenant_id"])  # any dim
    col.add(vectors, ids=ids, metadata=metadata)
    hits = col.search(qvec, where={"tenant_id": "acme"}, top_k=10)

The client holds no engine code and needs no numpy — vectors may be numpy arrays or plain lists.
"""
from __future__ import annotations

import socket
import threading

from telys._wire import recv_msg, send_msg


class RemoteError(RuntimeError):
    """An error raised by the server while handling a request (carries the server-side type + message)."""


def _listify(v):
    """Deep-coerce to plain JSON-able values, without importing numpy: numpy arrays AND scalars (np.float32,
    np.int64, …) both expose .tolist(); dict values (e.g. metadata pulled from a dataframe) are coerced too,
    since json.dumps rejects numpy scalars."""
    if hasattr(v, "tolist"):          # numpy ndarray OR scalar -> python list/number
        return v.tolist()
    if isinstance(v, dict):
        return {k: _listify(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_listify(x) for x in v]
    return v


def _parse_addr(address: str):
    """Resolve an address to ('unix', path) or ('tcp', (host, port)). Accepts:
    'unix:///run/telys.sock', '/abs/path.sock', './rel.sock', 'tcp://host:port', 'host:port'."""
    if address.startswith("unix://"):
        return "unix", address[len("unix://"):]
    if address.startswith("tcp://"):
        address = address[len("tcp://"):]
    elif "/" in address or address.startswith("."):     # looks like a filesystem path
        return "unix", address
    if ":" in address:
        host, _, port = address.rpartition(":")
        return "tcp", (host or "127.0.0.1", int(port))
    return "unix", address


class RemoteTelys:
    """Client handle to a self-hosted Telys server (one shared engine). ``address`` is a Unix socket path or
    a 'host:port' / 'tcp://host:port'; ``token`` is the shared access token for a TCP server."""

    def __init__(self, address: str, token: str | None = None, *, connect_timeout: float = 30.0) -> None:
        self._addr = address
        kind, target = _parse_addr(address)
        fam = socket.AF_UNIX if kind == "unix" else socket.AF_INET
        self._sock = socket.socket(fam, socket.SOCK_STREAM)
        # Bound connect + handshake so a dead/hostile server can't hang the client forever; then go blocking
        # for data ops (which may legitimately take a while).
        self._sock.settimeout(connect_timeout)
        self._sock.connect(target)
        self._lock = threading.Lock()  # one in-flight request/response per connection
        if token is not None:          # auth handshake (TCP); must precede any other op
            try:
                with self._lock:
                    send_msg(self._sock, {"op": "auth", "collection": None, "args": {"token": token}})
                    reply = recv_msg(self._sock)
            except (OSError, socket.timeout) as exc:
                self._sock.close()
                raise RemoteError(f"auth handshake failed: {exc}") from exc
            if not reply.get("ok"):
                self._sock.close()
                raise RemoteError(reply.get("error", "authentication failed"))
        self._sock.settimeout(None)

    def _call(self, op: str, collection: str | None = None, **args):
        with self._lock:
            send_msg(self._sock, {"op": op, "collection": collection, "args": args})
            reply = recv_msg(self._sock)
        if not reply.get("ok"):
            raise RemoteError(reply.get("error", "unknown server error"))
        return reply.get("result")

    # ── engine surface ───────────────────────────────────────────────────────────────────────────────
    def ping(self) -> bool:
        return bool(self._call("ping").get("pong"))

    def create_collection(self, name: str, dim: int, partition_by, *, filter_columns=(),
                          dtype: str = "f32", embedder: str | None = None) -> "RemoteCollection":
        self._call("create_collection", name, dim=dim, partition_by=partition_by,
                   filter_columns=list(filter_columns), dtype=dtype, embedder=embedder)
        return RemoteCollection(self, name)

    def open_collection(self, name: str) -> "RemoteCollection":
        self._call("open_collection", name)
        return RemoteCollection(self, name)

    def drop_collection(self, name: str) -> None:
        """Delete a collection (and its on-disk data) from the shared server. No-op if it doesn't exist."""
        self._call("drop_collection", name)

    def collections(self) -> list:
        return self._call("collections")

    def __getitem__(self, name: str) -> "RemoteCollection":
        return RemoteCollection(self, name)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RemoteCollection:
    """Mirrors telys.engine.Collection — forwards each op to the shared server."""

    def __init__(self, engine: RemoteTelys, name: str) -> None:
        self._engine = engine
        self.name = name

    def add(self, vectors, ids, metadata):
        return self._engine._call("add", self.name, vectors=_listify(vectors),
                                  ids=_listify(ids), metadata=_listify(metadata))

    def upsert(self, vectors, ids, metadata):
        return self._engine._call("upsert", self.name, vectors=_listify(vectors),
                                  ids=_listify(ids), metadata=_listify(metadata))

    def add_texts(self, texts, ids, metadata):
        return self._engine._call("add_texts", self.name, texts=list(texts),
                                  ids=_listify(ids), metadata=_listify(metadata))

    def upsert_texts(self, texts, ids, metadata):
        return self._engine._call("upsert_texts", self.name, texts=list(texts),
                                  ids=_listify(ids), metadata=_listify(metadata))

    def search(self, vector, top_k: int = 10, where=None, explain: bool = False, with_metadata: bool = False):
        return self._engine._call("search", self.name, vector=_listify(vector), top_k=top_k,
                                  where=_listify(where), explain=explain, with_metadata=with_metadata)

    def search_text(self, text, top_k: int = 10, where=None, explain: bool = False, with_metadata: bool = False):
        return self._engine._call("search_text", self.name, text=text, top_k=top_k,
                                  where=_listify(where), explain=explain, with_metadata=with_metadata)

    def ids(self, where=None) -> list:
        return self._engine._call("ids", self.name, where=_listify(where))

    def delete(self, ids):
        return self._engine._call("delete", self.name, ids=_listify(ids))

    def compact(self):
        return self._engine._call("compact", self.name)

    def build_ivf(self, min_rows: int = 20000, target_recall: float = 0.98):
        """Build per-partition IVF for oversized partitions (Team/self-host parity with Collection.build_ivf).
        Exposed for benchmark 'optimize' phases that build the index between load and query."""
        return self._engine._call("build_ivf", self.name, min_rows=min_rows, target_recall=target_recall)

    def snapshot(self):
        return self._engine._call("snapshot", self.name)

    def save(self):
        return self._engine._call("save", self.name)

    def stats(self) -> dict:
        return self._engine._call("stats", self.name)


def connect(address: str, token: str | None = None) -> RemoteTelys:
    """Connect to a self-hosted Telys server. ``address`` is a Unix-socket path or 'host:port'/'tcp://host:port';
    pass ``token`` for a TCP server that requires a shared access token."""
    return RemoteTelys(address, token)
