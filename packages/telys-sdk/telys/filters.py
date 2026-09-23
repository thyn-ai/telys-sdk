"""Telys query filter types (public contract).

A filter is a small, pure value object the SDK and the runtime both understand. It carries no engine logic,
so it lives in the public SDK; the runtime imports it from here.
"""
from __future__ import annotations


class Eq:
    """Equality predicate for the natural query surface: search(q, k, where=Eq("tenant_id", "acme"))."""

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def __repr__(self):
        return f"{self.column} == {self.value!r}"
