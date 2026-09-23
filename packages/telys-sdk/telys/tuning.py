"""Telys tuning interfaces (public contract) — the pluggable, governed-optional tuner seam.

This module ships in the PUBLIC SDK and contains only the INTERFACE: the `Tuner` base, the auditable
`TuningPlan`, and small pure helpers. The default `HeuristicTuner` IMPLEMENTATION is part of the closed
runtime (telys-runtime) — `from telys.tuning import HeuristicTuner` resolves it lazily from the installed
runtime (D-30). The optional `telys-algenta` adapter ships `AlgentaTuner`, which also implements this `Tuner`.
A tuner output is always a `TuningPlan` (no hidden magic): the chosen physical knobs + expected metrics +
constraints + a decision trace.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

__all__ = ["Tuner", "TuningPlan", "HeuristicTuner"]


def _hash(obj) -> str:
    try:
        b = json.dumps(obj, sort_keys=True, default=str).encode()
    except TypeError:
        b = repr(obj).encode()
    return "sha256:" + hashlib.sha256(b).hexdigest()[:16]


def _k(x):
    return x.item() if hasattr(x, "item") else x


def _workload_digest(workload):
    """Source-free digest of a workload (boundary rule 3): collapse container values to their sizes so raw
    text/source never enters the plan's workload_hash. Shared by HeuristicTuner and the AlgentaTuner adapter."""
    if workload is None:
        return None
    if isinstance(workload, dict):
        return {k: (len(v) if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) else
                    (f"<{type(v).__name__}:{len(v)}>" if isinstance(v, (str, bytes)) else v))
                for k, v in sorted(workload.items())}
    return {"n": len(workload)} if hasattr(workload, "__len__") else {"workload": type(workload).__name__}


@dataclass
class TuningPlan:
    """An auditable tuning plan — what the tuner chose, why, and what it expects. Apply via collection.apply_tuning()."""
    collection: str
    tuner: str
    target_recall: float
    exact_crossover_rows: int
    ivf: dict                         # {enabled, min_rows, target_recall, partitions:[{key,rows,nlist,nprobe}]}
    expected: dict = field(default_factory=dict)     # {p50_ms, p95_ms, recall_at_k, fallback_rate}
    constraints: dict = field(default_factory=dict)
    decision_trace: list = field(default_factory=list)
    workload_hash: str = ""
    engine: str = "telys"
    plan_id: str = ""

    def __post_init__(self):
        if not self.plan_id:
            self.plan_id = "tune_" + _hash([self.collection, self.tuner, self.target_recall,
                                            self.exact_crossover_rows, self.ivf, self.workload_hash])[7:19]

    def as_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.as_dict(), **kw)


class Tuner:
    """Tuner interface. The SDK facade depends on THIS, never on a concrete tuner. Implementations:
    HeuristicTuner (telys-runtime, the zero-config default) and AlgentaTuner (telys-algenta adapter)."""
    name = "base"

    def tune_collection(self, collection, workload=None, objectives=None) -> TuningPlan:  # pragma: no cover
        raise NotImplementedError

    def choose_plan(self, stats, query):
        """Optional per-query override; None => use the engine's built-in partition-aware router."""
        return None

    def explain(self) -> dict:
        return {"tuner": self.name}


def __getattr__(name):
    # HeuristicTuner is a closed-runtime implementation; resolve it lazily so the public SDK module carries
    # no engine logic, while `from telys.tuning import HeuristicTuner` keeps working when the runtime is present.
    if name == "HeuristicTuner":
        try:
            from memengine.tuning_heuristic import HeuristicTuner
        except ImportError as e:  # noqa: BLE001
            raise ImportError(
                "HeuristicTuner is provided by the Telys runtime, which is not installed. "
                "Run `telys runtime install` (or pip install the telys-runtime package for local dev)."
            ) from e
        return HeuristicTuner
    raise AttributeError(f"module 'telys.tuning' has no attribute {name!r}")
