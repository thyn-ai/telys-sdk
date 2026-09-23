"""One-command MCP registration: `telys mcp install --client claude|cursor|codex|claude-desktop|qwen`.

Writes the canonical, ZERO-ENV entry — `{command: "telys", args: ["mcp"]}` — into each client's config, so an
enterprise user goes `pipx install telys → telys login → telys mcp install` and their assistant has Telys
memory. No PYTHONPATH, no kernel paths: the runtime self-locates from the verified `telys login` install.

Prefers a client's own CLI when it owns a file that also holds non-MCP state (Claude Code's ~/.claude.json,
Codex's config.toml); otherwise merges the MCP-only config file directly (idempotent, with a .bak backup).
Stdlib only — no third-party deps ship in the public SDK.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

_SERVER_ARGS = ["mcp"]


def _command() -> str:
    """The command a client should launch. Prefer bare `telys` (on PATH via pipx/pip); fall back to this
    interpreter's console script, then `python -m telys.cli`-style, so it works even off-PATH."""
    if shutil.which("telys"):
        return "telys"
    cand = os.path.join(os.path.dirname(sys.executable), "telys")
    return cand if os.path.exists(cand) else "telys"


def _entry_json(name: str, *, stdio_type: bool) -> dict:
    e = {"command": _command(), "args": list(_SERVER_ARGS)}
    if stdio_type:
        e = {"type": "stdio", **e}
    return e


def _backup(path: str) -> None:
    if os.path.exists(path):
        shutil.copy2(path, path + ".bak")


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh) or {}
    except (OSError, ValueError):
        return {}


def _merge_json_mcp(path: str, name: str, entry: dict, *, key: str = "mcpServers") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    data = _load_json(path)
    servers = data.get(key)
    if not isinstance(servers, dict):
        servers = {}
    servers[name] = entry
    data[key] = servers
    _backup(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def _remove_json_mcp(path: str, name: str, *, key: str = "mcpServers") -> bool:
    data = _load_json(path)
    servers = data.get(key)
    if isinstance(servers, dict) and name in servers:
        del servers[name]
        _backup(path)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        return True
    return False


# ── per-client config file locations ─────────────────────────────────────────────────────────────────────
def _home(*parts: str) -> str:
    return os.path.join(os.path.expanduser("~"), *parts)


def _claude_desktop_config() -> str:
    if sys.platform == "darwin":
        return _home("Library", "Application Support", "Claude", "claude_desktop_config.json")
    if sys.platform.startswith("win"):
        return os.path.join(os.environ.get("APPDATA", _home("AppData", "Roaming")), "Claude", "claude_desktop_config.json")
    return _home(".config", "Claude", "claude_desktop_config.json")


def _cursor_config(scope: str) -> str:
    return os.path.join(os.getcwd(), ".cursor", "mcp.json") if scope == "project" else _home(".cursor", "mcp.json")


def _claude_project_config() -> str:
    return os.path.join(os.getcwd(), ".mcp.json")


def _codex_config(scope: str) -> str:
    return os.path.join(os.getcwd(), ".codex", "config.toml") if scope == "project" else _home(".codex", "config.toml")


def _qwen_config(scope: str) -> str:
    return os.path.join(os.getcwd(), ".qwen", "settings.json") if scope == "project" else _home(".qwen", "settings.json")


def _detect_clients() -> list[str]:
    found = []
    if shutil.which("claude") or os.path.exists(_home(".claude.json")):
        found.append("claude")
    if os.path.isdir(_home(".cursor")):
        found.append("cursor")
    if os.path.exists(_claude_desktop_config()) or sys.platform == "darwin":
        found.append("claude-desktop")
    if shutil.which("codex") or os.path.isdir(_home(".codex")):
        found.append("codex")
    if os.path.isdir(_home(".qwen")):
        found.append("qwen")
    return found or ["claude"]


# ── install / uninstall per client ───────────────────────────────────────────────────────────────────────
def _install_claude(name: str, scope: str) -> str:
    # User scope: ~/.claude.json holds lots of non-MCP state — use the official CLI (owns the file). Project
    # scope: a clean MCP-only .mcp.json in the repo (safe to write directly).
    if scope == "user" and shutil.which("claude"):
        subprocess.run(["claude", "mcp", "add", "--transport", "stdio", "--scope", "user", name, "--", _command(), *_SERVER_ARGS], check=True)
        return "claude (user, via `claude mcp add`)"
    path = _claude_project_config()
    _merge_json_mcp(path, name, _entry_json(name, stdio_type=True))
    return f"claude (project) → {path}"


def _install_cursor(name: str, scope: str) -> str:
    path = _cursor_config(scope)
    _merge_json_mcp(path, name, _entry_json(name, stdio_type=False))
    return f"cursor ({scope}) → {path}"


def _install_claude_desktop(name: str, scope: str) -> str:
    path = _claude_desktop_config()
    _merge_json_mcp(path, name, _entry_json(name, stdio_type=False))
    return f"claude-desktop → {path} (restart Claude Desktop)"


def _install_codex(name: str, scope: str) -> str:
    if shutil.which("codex"):
        subprocess.run(["codex", "mcp", "add", name, "--", _command(), *_SERVER_ARGS], check=True)
        return "codex (via `codex mcp add`)"
    # Minimal TOML merge (one flat table) — avoids a tomli-w dep.
    path = _codex_config(scope)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _backup(path)
    existing = ""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            existing = fh.read()
    block = (f'\n[mcp_servers.{name}]\ncommand = "{_command()}"\nargs = ['
             + ", ".join(f'"{a}"' for a in _SERVER_ARGS) + "]\n")
    if f"[mcp_servers.{name}]" in existing:
        return f"codex → {path} (already present; left as-is)"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(block)
    return f"codex → {path}"


def _install_qwen(name: str, scope: str) -> str:
    # settings.json carries lots of non-MCP state and the format allows JSONC comments — refuse to touch a
    # file that doesn't parse as strict JSON. The generic merge path recovers from parse errors by starting
    # from {}, which would CLOBBER the user's whole settings file here; fail closed instead.
    path = _qwen_config(scope)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                json.load(fh)
        except ValueError as exc:
            raise RuntimeError(
                f"{path} is not strict JSON (JSONC comments?) — add the mcpServers entry manually"
            ) from exc
    _merge_json_mcp(path, name, _entry_json(name, stdio_type=False))
    return f"qwen ({scope}) → {path} (restart Qwen Code)"


_INSTALLERS = {
    "claude": _install_claude,
    "cursor": _install_cursor,
    "claude-desktop": _install_claude_desktop,
    "codex": _install_codex,
    "qwen": _install_qwen,
}


def install(*, client: str = "all", scope: str = "user", name: str = "telys") -> int:
    clients = _detect_clients() if client == "all" else [client]
    print(f"registering the Telys MCP server ({_command()} {' '.join(_SERVER_ARGS)}) — no env, self-locating runtime")
    rc = 0
    for c in clients:
        fn = _INSTALLERS.get(c)
        if not fn:
            print(f"  [skip] unknown client: {c}"); rc = 1; continue
        try:
            print(f"  ✓ {fn(name, scope)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {c}: {exc}"); rc = 1
    print("done. Restart/reload the client, then ask it to use `telys_search` / `telys_add`.")
    print("(Telys memory needs the runtime — run `telys login` once if you haven't.)")
    return rc


def uninstall(*, client: str = "all", scope: str = "user", name: str = "telys") -> int:
    clients = _detect_clients() if client == "all" else [client]
    for c in clients:
        if c == "claude" and scope == "user" and shutil.which("claude"):
            subprocess.run(["claude", "mcp", "remove", name], check=False); print(f"  ✓ claude (user): removed {name}"); continue
        paths = {"claude": _claude_project_config(), "cursor": _cursor_config(scope),
                 "claude-desktop": _claude_desktop_config(), "codex": _codex_config(scope),
                 "qwen": _qwen_config(scope)}
        p = paths.get(c)
        if c == "codex":
            print(f"  [manual] remove the [mcp_servers.{name}] block from {p}"); continue
        print(f"  {'✓ removed from' if p and _remove_json_mcp(p, name) else '· not present in'} {p}")
    return 0


def status(*, name: str = "telys") -> int:
    print(f"Telys MCP command: {_command()} {' '.join(_SERVER_ARGS)}")
    checks = {"claude (project)": _claude_project_config(), "cursor (user)": _cursor_config("user"),
              "claude-desktop": _claude_desktop_config(), "codex (user)": _codex_config("user"),
              "qwen (user)": _qwen_config("user")}
    for label, path in checks.items():
        present = name in (_load_json(path).get("mcpServers", {}) if path.endswith(".json") else {})
        if path.endswith(".toml"):
            present = os.path.exists(path) and f"[mcp_servers.{name}]" in open(path, encoding="utf-8").read()
        print(f"  {'registered' if present else 'not registered':14} {label}: {path}")
    return 0
