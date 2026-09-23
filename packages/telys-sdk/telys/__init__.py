"""Telys — embedded, on-device memory & retrieval SDK (public, thin).

    from telys import Telys, Eq, scope_key
    eng = Telys(path, embedding_providers={...})
    eng.create_collection(name, dim, partition_by, embedder=, dtype=, filter_columns=)
    eng.open_collection(name, embedder=) · eng.collections()
    col.add_texts / add / upsert_texts / upsert
    col.search_text / search (where=, top_k=, explain=, target_recall=, with_metadata=)
    col.ids(where=)   # live external ids in a scope (or off-key column)
    col.delete / update_texts / compact / build_ivf / save / stats / snapshot

This is the public SDK: facades, types, and provider/tuner interfaces only — no engine implementation. The
engine is a separate, closed, on-device runtime loaded on first use (`telys runtime install`; see D-30).
`telys turns filtered vector search into a contiguous memory operation.`
"""
def _activate_installed_runtime() -> None:
    """Zero-config runtime: after a VERIFIED `telys runtime install`, add the installed runtime's Python package
    dir (pysite, from the signed bundle's slim telys-runtime-native wheel) to sys.path — so the native engine and
    kernel-backed embedders import with NO env vars. Only ever prepends a telys-managed, verified path; a silent
    no-op when nothing is installed.

    Ambient-engine-wins rule: pysite is prepended ONLY when no engine package (`memengine`) is already
    resolvable in this environment. An ambient `memengine` — an editable dev install of `telys-runtime`, a
    pip-installed `telys-runtime-native` (any version), or a source tree a harness put on sys.path — is an
    explicit environment choice and must never be shadowed by the bundle's slim shim: the shim is the
    native-only outsider artifact and intentionally lacks the internal reference engine
    (`memengine.runtime`, `memengine.tuning_heuristic`, ...), so shadowing breaks dev environments and tests
    the moment ANY runtime is installed. (Supersedes the b4 version guard, which compared telys-runtime-native
    versions only and could not see the fat editable `telys-runtime` at all — that guard's b3-over-b1 case is
    the strict subset where an ambient wheel provides `memengine`.)"""
    try:
        import json
        import os
        import sys

        import telys.paths as _paths

        record = os.path.join(_paths.install_dir(), _paths.INSTALL_RECORD_NAME)
        with open(record, encoding="utf-8") as fh:
            if not json.load(fh).get("verified"):
                return
        site = _paths.installed_pysite()
        if not site or site in sys.path:
            return
        if _ambient_engine_present():
            return
        sys.path.insert(0, site)
    except Exception:  # noqa: BLE001 — never let runtime activation break `import telys`
        return


def _ambient_engine_present() -> bool:
    """True iff the engine package (`memengine`) is ALREADY resolvable in this environment, WITHOUT importing
    it (find_spec resolves without executing module code, so this has no engine side effects)."""
    try:
        import importlib.util

        return importlib.util.find_spec("memengine") is not None
    except Exception:  # noqa: BLE001 — resolution probe must never break `import telys`
        return False


_activate_installed_runtime()

from telys.engine import AME, Telys, FORMAT_VERSION, __version__, scope_key  # noqa: F401,E402
from telys.filters import Eq  # noqa: F401,E402
from telys.tuning import Tuner, TuningPlan  # noqa: F401,E402  (HeuristicTuner is lazy — see __getattr__)

__all__ = ["Telys", "AME", "Eq", "scope_key", "Tuner", "HeuristicTuner", "TuningPlan",
           "__version__", "FORMAT_VERSION"]


def __getattr__(name):
    # HeuristicTuner is a closed-runtime implementation; expose it lazily so `import telys` never loads the
    # engine, while `from telys import HeuristicTuner` keeps working when the runtime is installed.
    if name == "HeuristicTuner":
        from telys.tuning import HeuristicTuner
        return HeuristicTuner
    raise AttributeError(f"module 'telys' has no attribute {name!r}")
