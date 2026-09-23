"""Tiny length-prefixed JSON wire protocol shared by the Telys self-host server + client.

Frame = 4-byte big-endian unsigned length, then that many bytes of UTF-8 JSON. A request is
``{"op": str, "collection": str|null, "args": {...}}``; a reply is ``{"ok": true, "result": ...}`` or
``{"ok": false, "error": str}``. Vectors travel as plain JSON number lists (MVP; a binary frame is a later
optimization). The 4-byte length is bounded to fail closed on a hostile/oversized frame (anti-OOM).
"""
from __future__ import annotations

import json
import socket
import struct

MAX_FRAME = 256 * 1024 * 1024  # 256 MiB hard ceiling per message (anti-OOM on a bad/hostile length prefix)


def send_msg(sock: socket.socket, obj) -> None:
    data = json.dumps(obj).encode("utf-8")
    if len(data) > MAX_FRAME:
        raise ValueError(f"frame too large: {len(data)} > {MAX_FRAME}")
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed the connection mid-frame")
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket):
    (n,) = struct.unpack(">I", _recv_exact(sock, 4))
    if n > MAX_FRAME:
        raise ValueError(f"declared frame too large: {n} > {MAX_FRAME}")
    return json.loads(_recv_exact(sock, n).decode("utf-8"))
