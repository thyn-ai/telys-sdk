"""Telys embedding interfaces (public contract) — embedding-agnostic by design (D-20).

Telys does NOT compete on embedding. The index ingests float32 vectors from ANY source; an embedder is an
optional, pluggable provider, and the engine is fully usable without one. This module ships in the PUBLIC SDK
and contains only the INTERFACE:
  - EmbeddingProfile  : immutable identity of a vector SPACE (same dim != same space).
  - EmbeddingProvider : the tiny Protocol the engine depends on (profile, embed_documents, embed_queries).
  - CallableEmbedder  : adapt any external provider (OpenAI/Cohere/BGE/E5/ONNX/...) via a callable + profile.

The lexical kernel embedders (AlgentaBigramEmbedder, AlgentaMultigramEmbedder) are thin ctypes BINDINGS over
the installed native kernel (libame_kernel). D-30 refinement: shipping the kernel *binding* in the public SDK
is fine — the moat is the COMPILED hashing/logic inside the Mojo kernel, not this ~30-line glue. So they work
zero-config right after `telys runtime install` / `telys login` (the signed kernel is already on disk), with no
separate runtime Python package and no env vars. AlgentaPooledEmbedder (needs an external token table) stays in
the closed runtime and is resolved lazily. The kernel binary is NEVER embedded here — only bound at call time.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["EmbeddingProvider", "EmbeddingProfile", "CallableEmbedder",
           "AlgentaBigramEmbedder", "AlgentaMultigramEmbedder", "AlgentaPooledEmbedder"]


@dataclass(frozen=True)
class EmbeddingProfile:
    """Immutable identity of a vector space. Two profiles are interchangeable IFF their space_id matches."""
    provider: str
    model_id: str
    model_version: str
    dimension: int
    dtype: str = "float32"
    normalization: str = "l2"        # "l2" | "none"
    distance: str = "ip"             # "ip" (cosine when L2-normalized) | "l2"
    pooling: str = "none"            # "none" | "bigram-hash" | "masked-mean" | ...
    tokenizer_hash: str = ""
    projection_version: str = ""
    created: str = ""                # caller-stamped ISO timestamp (engine does not call the clock)

    def space_id(self) -> str:
        ident = "|".join(str(x) for x in (
            self.provider, self.model_id, self.model_version, self.dimension, self.dtype,
            self.normalization, self.distance, self.pooling, self.tokenizer_hash, self.projection_version))
        return "sha256:" + hashlib.sha256(ident.encode()).hexdigest()[:16]

    def compatible_with(self, other: "EmbeddingProfile") -> bool:
        return self.space_id() == other.space_id()

    def as_dict(self) -> dict:
        d = asdict(self); d["space_id"] = self.space_id(); return d


class EmbeddingProvider:
    """Protocol the engine depends on. Implementations return float32 [N, D] arrays."""
    @property
    def profile(self) -> EmbeddingProfile:  # pragma: no cover - interface
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)


class CallableEmbedder(EmbeddingProvider):
    """Adapt ANY external provider: pass a callable (list[str] -> [N,D] float32) + a declared profile.

    Keeps Telys free of OpenAI/Cohere/HF/ONNX dependencies — the caller owns those:
        CallableEmbedder(lambda ts: st_model.encode(ts, normalize_embeddings=True),
                         EmbeddingProfile("hf", "bge-small-en", "1.5", 384, distance="ip"))
    """

    def __init__(self, fn, profile: EmbeddingProfile, query_fn=None) -> None:
        self._fn = fn
        self._qfn = query_fn or fn
        self._profile = profile

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        v = np.ascontiguousarray(self._fn(list(texts)), np.float32)
        assert v.ndim == 2 and v.shape[1] == self._profile.dimension, "provider output shape != profile.dimension"
        return v

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return np.ascontiguousarray(self._qfn(list(texts)), np.float32)


# ── native kernel binding (libame_kernel) ────────────────────────────────────────────────────────────────
_INT32_MIN, _INT32_MAX = -(2**31), 2**31 - 1


def _fits_i32(n: int, what: str = "count") -> int:
    """A length passed to the C int32 kernel args must fit int32 (a >2^31 value wraps negative → kernel UB)."""
    n = int(n)
    if n < _INT32_MIN or n > _INT32_MAX:
        raise OverflowError(f"{what}={n} does not fit int32 [{_INT32_MIN}, {_INT32_MAX}]")
    return n


_KERNEL = None


def _kernel():
    """Load libame_kernel + register the embed symbols. Resolution (zero-config after `telys runtime install`):
    TELYS_KERNEL / AME_KERNEL env → the verified install ($TELYS_HOME, via telys.paths) → in-tree mojo_build
    (dev). Cached. Raises a clear, actionable ImportError if the signed kernel isn't installed."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    from telys import paths

    p = os.environ.get("TELYS_KERNEL") or os.environ.get("AME_KERNEL")
    if not p:
        p = paths.installed_lib_path("libame_kernel")  # $TELYS_HOME/runtime/<tag>/libame_kernel.<ext> if present
    if not p:
        here = os.path.dirname(os.path.abspath(__file__))
        dev = os.path.abspath(os.path.join(here, "..", "..", "..", "mojo_build", f"libame_kernel.{paths.lib_ext()}"))
        p = dev if os.path.exists(dev) else None
    if not p or not os.path.exists(p):
        raise ImportError(
            "the Telys on-device kernel (libame_kernel) is not installed. Run `telys login` (or "
            "`telys runtime install`) to fetch the signed runtime for your platform. For a BYO-model embedder "
            "with no kernel, use telys.embedding.CallableEmbedder."
        )
    lib = ctypes.CDLL(p)
    for _dimfn in ("ame_embed_unigram_dim", "ame_embed_bigram_dim", "ame_embed_trigram_dim", "ame_embed_multigram_dim"):
        fn = getattr(lib, _dimfn)
        fn.restype = ctypes.c_int32
        fn.argtypes = []
    lib.ame_embed_bigram.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]      # text, n, out[dim]
    lib.ame_embed_multigram.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]   # text, n, out[dim]
    _KERNEL = lib
    return lib


class AlgentaBigramEmbedder(EmbeddingProvider):
    """Fast lexical bigram embedder (kernel ame_embed_bigram). Typo-tolerant, zero network, zero model files.
    The dim is sourced from the kernel (ame_embed_bigram_dim) — never hardcoded. Zero-vector for empty input
    (sorts last under inner product); never L2-renormalize it."""

    def __init__(self, *, max_bytes: int = 1 << 20) -> None:
        self.lib = _kernel()
        self.dim = int(self.lib.ame_embed_bigram_dim())
        if self.dim <= 0:
            raise RuntimeError(f"kernel reported a non-positive bigram dim ({self.dim})")
        self.max_bytes = int(max_bytes)

    @property
    def profile(self) -> EmbeddingProfile:
        return EmbeddingProfile("algenta", "bigram", "1.0.0", self.dim, normalization="l2", distance="ip",
                                pooling="bigram-hash", tokenizer_hash="utf8-bytes")

    def _one(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError(f"bigram embedder expects str, got {type(text).__name__}")
        b = text.encode("utf-8")
        if b"\x00" in b:
            raise ValueError("text contains an embedded NUL byte (U+0000) — not allowed")
        if len(b) > self.max_bytes:
            b = b[: self.max_bytes]
        _fits_i32(len(b), "text byte length")
        o = np.zeros(self.dim, np.float32)
        self.lib.ame_embed_bigram(b, len(b), o.ctypes.data_as(ctypes.c_void_p))
        assert o.shape[0] == self.dim
        return o

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.ascontiguousarray(np.stack([self._one(t) for t in texts]), np.float32)


class AlgentaMultigramEmbedder(EmbeddingProvider):
    """Lexical MULTIGRAM fusion embedder (kernel ame_embed_multigram): concatenated [unigram|bigram|trigram],
    each block L2-normalised independently (so inner product = cos_u+cos_b+cos_t). Typo-tolerant, zero network,
    zero model artifacts. Every dim is sourced from the kernel; zero-vector contract is PER BLOCK (never
    L2-renormalize the whole vector)."""

    def __init__(self, *, max_bytes: int = 1 << 20) -> None:
        self.lib = _kernel()
        self.dim = int(self.lib.ame_embed_multigram_dim())
        self.block_dims = (int(self.lib.ame_embed_unigram_dim()),
                           int(self.lib.ame_embed_bigram_dim()),
                           int(self.lib.ame_embed_trigram_dim()))
        if self.dim <= 0 or sum(self.block_dims) != self.dim:
            raise RuntimeError(f"kernel multigram layout inconsistent: dim={self.dim} blocks={self.block_dims}")
        self.max_bytes = int(max_bytes)

    @property
    def profile(self) -> EmbeddingProfile:
        return EmbeddingProfile("algenta", "multigram", "1.0.0", self.dim, normalization="l2-blockwise",
                                distance="ip", pooling="multigram-hash", tokenizer_hash="utf8-bytes")

    def _one(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError(f"multigram embedder expects str, got {type(text).__name__}")
        b = text.encode("utf-8")
        if b"\x00" in b:
            raise ValueError("text contains an embedded NUL byte (U+0000) — not allowed")
        if len(b) > self.max_bytes:
            b = b[: self.max_bytes]
        _fits_i32(len(b), "text byte length")
        o = np.zeros(self.dim, np.float32)
        self.lib.ame_embed_multigram(b, len(b), o.ctypes.data_as(ctypes.c_void_p))
        assert o.shape[0] == self.dim
        return o

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.ascontiguousarray(np.stack([self._one(t) for t in texts]), np.float32)


def __getattr__(name):
    # AlgentaPooledEmbedder needs an external token-embedding table + more kernel surface, so it stays in the
    # closed runtime and resolves lazily. (Bigram/Multigram now bind the kernel directly, above.)
    if name == "AlgentaPooledEmbedder":
        try:
            import memengine.embedding as _e
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "AlgentaPooledEmbedder is provided by the Telys runtime, which is not installed. "
                "Run `telys runtime install`. For a BYO-model embedder use telys.embedding.CallableEmbedder."
            ) from exc
        return getattr(_e, name)
    raise AttributeError(f"module 'telys.embedding' has no attribute {name!r}")
