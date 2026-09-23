"""Telys SDK runtime boundary — interface + loader (public).

The public SDK facade talks ONLY to a `RuntimeHandle`. The interface lives here (public contract); the
implementation (`LocalRuntime`, wrapping the in-process engine today; a native runtime in a later sub-phase)
ships in the closed `telys-runtime` package and is resolved lazily by `load_runtime()`. `import telys` never
imports the engine — it is loaded the first time you create/open a collection (D-30).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Public on-disk format contract. The runtime asserts its engine matches this; bump deliberately.
FORMAT_VERSION = 1


class RuntimeHandle:
    """Stable interface between the Telys SDK facade and the engine. All ops take the opaque index handle as
    the first argument (returned by new_index/open_index; the facade never introspects it)."""

    FORMAT_VERSION = FORMAT_VERSION

    # lifecycle
    def new_index(self, dim, dtype): raise NotImplementedError
    def open_index(self, path): raise NotImplementedError
    def save(self, index, path): raise NotImplementedError
    def sealed(self, index) -> bool: raise NotImplementedError

    # ingest
    def attach_embedder(self, index, embedder): raise NotImplementedError
    def build(self, index, vecs, iids, keys, key_name, columns, texts=None): raise NotImplementedError
    def insert(self, index, vecs, iids, keys, columns, texts=None): raise NotImplementedError
    def update(self, index, iids, vecs, keys, columns, texts=None): raise NotImplementedError

    # query
    def search(self, index, vector, top_k, where, explain, target_recall): raise NotImplementedError
    def columns_for(self, index, ids): raise NotImplementedError
    def live_ids(self, index, where): raise NotImplementedError

    # lexical (on-device BM25 code index; build after ingest, query with mode="lexical")
    def build_lexical(self, index, k1=1.8, b=1.0): raise NotImplementedError
    def search_lexical(self, index, text, top_k, where, explain): raise NotImplementedError

    # maintain
    def delete(self, index, iids): raise NotImplementedError
    def compact(self, index): raise NotImplementedError
    def snapshot(self, index): raise NotImplementedError
    def stats(self, index) -> dict: raise NotImplementedError

    # compact serve artifacts (ultra-small READ-ONLY export: quantized slab + remap + provenance
    # columns; no f32 base, no external id map, no lexical tokens — the "ship the index" tier)
    def export_compact(self, index, path, mode): raise NotImplementedError
    def open_compact(self, path): raise NotImplementedError

    # physical tuning state (driven by Collection.apply_tuning; max_nprobe exposed for governed tuners)
    def build_partition_ivf(self, index, min_rows, target_recall, max_nprobe=64): raise NotImplementedError
    def set_exact_crossover(self, index, rows): raise NotImplementedError
    def partition_ivf_keys(self, index): raise NotImplementedError
    def set_partition_nprobe(self, index, key, nprobe): raise NotImplementedError

    # default zero-config tuner (HeuristicTuner — a closed-runtime impl)
    def default_tuner(self): raise NotImplementedError

    # diagnostics — where the native kernel resolves from (so the SDK/CLI never imports engine internals)
    def kernel_info(self) -> dict: raise NotImplementedError


class RuntimeNotInstalled(RuntimeError):
    """Raised when the Telys engine runtime is not available. Install it with `telys runtime install`."""


def _under_install_cache(path: str) -> bool:
    """True iff `path` resolves to a location inside the verified install cache ($TELYS_HOME/runtime/<platform>).

    Uses strict resolution + `relative_to` so a symlink cannot spoof containment."""
    from telys import paths

    try:
        child = Path(path).resolve(strict=True)
        root = Path(paths.install_dir()).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return False
    try:
        child.relative_to(root)
        return True
    except ValueError:
        return False


def _verified_install_present() -> bool:
    """True when `telys runtime install` has placed a VERIFIED native runtime under the install cache.

    This is the untrusted-safe path: the native lib resolves from inside the cache and the install record marks
    it verified. Such installs are native-only — the readable-Python LocalRuntime must never serve them."""
    from telys import paths

    lib = paths.installed_lib_path(paths.TY_RUNTIME_STEM)
    if not lib or not _under_install_cache(lib):
        return False
    record = os.path.join(paths.install_dir(), paths.INSTALL_RECORD_NAME)
    try:
        with open(record, encoding="utf-8") as fh:
            return bool(json.load(fh).get("verified"))
    except (OSError, ValueError):
        return False


def load_runtime() -> RuntimeHandle:
    """Resolve and instantiate the Telys runtime. Lazy: importing `telys` does not import the engine.

    TELYS_RUNTIME forces the implementation: 'native' (ctypes-bound native runtime) or 'local' (in-process
    Python over the kernel). With no override, a VERIFIED install ($TELYS_HOME/runtime, from `telys runtime
    install`) is served **native-only and fail-closed** — the readable-Python LocalRuntime is never a fallback
    there, so an outsider install cannot silently execute engine source. Dev/internal editable installs (e.g.
    Codna) keep the default local path unchanged."""
    which = (os.environ.get("TELYS_RUNTIME") or "").lower()
    verified = which == "" and _verified_install_present()
    if which == "native" or verified:
        try:
            from memengine.native_runtime import NativeRuntime
        except ImportError as e:  # noqa: BLE001
            extra = (
                " A verified runtime is installed but its native engine could not be loaded; the readable-Python "
                "fallback is refused for verified installs (reinstall with `telys runtime install`)."
                if verified else
                " Run `telys runtime install` (or `pip install telys-runtime` for local dev)."
            )
            raise RuntimeNotInstalled(f"The native Telys runtime is unavailable.{extra}") from e
        return NativeRuntime()
    try:
        from memengine.runtime import LocalRuntime
    except ImportError as e:  # noqa: BLE001
        raise RuntimeNotInstalled(
            "The Telys runtime is not installed. Run `telys runtime install` to fetch the signed on-device "
            "runtime, or `pip install telys-runtime` for local development."
        ) from e
    return LocalRuntime()


def runtime_available() -> bool:
    """True if a Telys runtime can be loaded. A verified zero-config install serves the NATIVE runtime
    (`memengine.native_runtime` — the only engine module the slim `telys-runtime-native` wheel ships), while
    dev/editable installs additionally carry the readable `LocalRuntime` (`memengine.runtime`). Either counts,
    so `telys runtime status` recognises a real signed install and not merely a dev checkout. (Probing only
    `memengine.runtime` reported NOT INSTALLED on every verified zero-config install, since that module is
    intentionally excluded from the outsider-safe wheel.)"""
    for mod in ("memengine.native_runtime", "memengine.runtime"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False
