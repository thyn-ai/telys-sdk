"""Telys CLI — `telys ...` (scaffold).

Commands:
  telys login                 authenticate for gated runtime downloads (Phase 1: real auth)
  telys runtime status        show whether the on-device runtime is installed + where it resolves from
  telys runtime install --file <bundle> --license <license.jwt>
                              verify a signed runtime bundle + offline RS256 license LOCALLY and install it
                              (offline-file install; hosted packages.telys.ai download lands later — D-31 #4)
  telys runtime verify        re-verify the installed runtime offline (manifest signature + artifact SHA-256 + license)
  telys runtime update        update to the latest runtime on the configured channel (Phase 1)
  telys version               print SDK version + format version

The SDK is thin: `runtime install` is the ONLY step that may touch the network (and offline-file install does
not even do that). Verification + query execution are always local/offline (D-30/D-31): the runtime + license
verify themselves against Telys public keys with no network, no OS-native signing.
"""
from __future__ import annotations

import argparse
import os
import sys


def _runtime_status() -> int:
    from telys.runtime import runtime_available, load_runtime
    from telys import __version__, FORMAT_VERSION
    if not runtime_available():
        print("runtime: NOT INSTALLED")
        print("  install it:  telys runtime install   (or, for local dev:  pip install telys-runtime)")
        return 1
    # Ask the runtime where its kernel resolves from — the SDK never imports engine internals (D-30).
    info = load_runtime().kernel_info()
    ok = bool(info.get("found"))
    if ok:
        print(f"runtime: READY  (telys {__version__}, format v{FORMAT_VERSION}, impl={info.get('runtime')})")
        print(f"  kernel: FOUND  [{info.get('source')}]  {info.get('path')}")
        return 0
    print(f"runtime: PACKAGE PRESENT, KERNEL NOT INSTALLED  (telys {__version__}, impl={info.get('runtime')})")
    print(f"  kernel: MISSING  [{info.get('source')}]  {info.get('path')}")
    print("  install it:  telys runtime install --file <bundle> --license <license.jwt>")
    return 1


def _not_implemented(what: str) -> int:
    print(f"`telys {what}` is not implemented yet (Phase 1: gated download + signature verification).")
    print("For local development, install the runtime package directly:  pip install telys-runtime")
    return 2


def _runtime_install(args) -> int:
    # D-31 #4: --file = offline-file install (no network). Otherwise download the signed bundle from the host
    # (the ONLY networked step) and verify it LOCALLY — same RS256/SHA-256 gate either way.
    from telys.installer import install_from_file, install_from_host, InstallError
    from telys.verify import VerificationError
    try:
        if getattr(args, "file", None):
            rec = install_from_file(args.file, args.license)
        else:
            from telys.paths import packages_url
            print(f"downloading runtime '{args.version}' from {packages_url()} …")
            rec = install_from_host(version=args.version, license_path=getattr(args, "license", None))
    except (InstallError, VerificationError) as e:
        print(f"install FAILED: {e}")
        return 1
    lic = rec.get("license", {})
    print(f"runtime INSTALLED + VERIFIED  [{rec['platform']}]  ({len(rec['artifacts'])} artifacts)")
    print(f"  license: tier={lic.get('tier')} features={lic.get('features')} exp={lic.get('exp')}")
    if rec.get("source"):
        print(f"  source: {rec['source']}")
    print("  query execution is now fully local/offline — no network.")
    return 0


def _login(args) -> int:
    # One command: OAuth device-code -> create API key -> register device -> platform-signed license ->
    # install + verify the signed runtime -> cache. After this, Telys runs fully offline. --token skips the
    # browser step (headless/CI); --no-install provisions without downloading the runtime.
    from telys.login import login, LoginError
    try:
        info = login(
            plan=getattr(args, "plan", "telys_developer"),
            access_token=getattr(args, "token", None) or os.environ.get("TELYS_TOKEN"),
            install=not getattr(args, "no_install", False),
            open_browser=not getattr(args, "no_browser", False),
        )
    except LoginError as e:
        print(f"login FAILED: {e}")
        return 1
    runtime = "installed" if info["installed"] else "not installed (run `telys runtime install`)"
    print(f"  api key: {info['api_key_prefix']}   device: {info['device_id'][:12]}…   "
          f"tier: {info['tier']}   runtime: {runtime}")
    return 0


def _mem(args) -> int:
    import json as _json

    from telys.mcp import TelysMemory

    m = TelysMemory(getattr(args, "path", None))
    try:
        if args.memcmd == "list":
            out = m.list_collections()
        elif args.memcmd == "create":
            out = m.create_collection(args.name, partition_by=args.partition_by)
        elif args.memcmd == "add":
            out = m.add(args.collection, texts=args.text, ids=args.id or None)
        elif args.memcmd == "search":
            out = m.search(args.collection, args.query, top_k=args.top_k)
        elif args.memcmd == "stats":
            out = m.stats(args.collection)
        else:
            print("usage: telys mem {add,search,create,list,stats} …")
            return 2
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}")
        return 1
    print(_json.dumps(out, indent=2))
    return 0


def _runtime_verify() -> int:
    from telys.installer import verify_installed, InstallError
    from telys.verify import VerificationError
    try:
        rep = verify_installed()
    except (InstallError, VerificationError) as e:
        print(f"runtime verify FAILED: {e}")
        return 1
    print(f"runtime VERIFIED  [{rep['platform']}]  {rep['dir']}")
    print(f"  artifacts: {', '.join(rep['artifacts_verified'])}")
    print(f"  license: tier={rep.get('tier')} exp={rep.get('exp')}")
    return 0


def _serve(args) -> int:
    import os
    license_token = args.license or os.environ.get("TELYS_LICENSE")
    access_token = args.access_token or os.environ.get("TELYS_ACCESS_TOKEN")
    if not args.socket and not (args.host and args.port):
        print("error: pass --socket <path> (Unix) or --host <h> --port <p> (TCP)")
        return 2

    if args.supervise:
        # Re-run ourselves as a supervised child WITHOUT --supervise. Secrets go via env (inherited by the
        # child), never on the child's argv, so they don't leak into the process list.
        from telys._supervisor import supervise
        if license_token:
            os.environ["TELYS_LICENSE"] = license_token
        if access_token:
            os.environ["TELYS_ACCESS_TOKEN"] = access_token
        child = ["serve", "--path", args.path]
        if args.socket:
            child += ["--socket", args.socket]
        if args.host:
            child += ["--host", args.host]
        if args.port:
            child += ["--port", str(args.port)]
        if args.require_license:
            child += ["--require-license"]
        if args.save_interval:
            child += ["--save-interval", str(args.save_interval)]
        print(f"telys serve (supervised): {args.path}")
        return supervise(child)

    from telys.server import serve
    where = args.socket if args.socket else f"{args.host}:{args.port}"
    print(f"telys serve: shared memory at {args.path} on {where}"
          + ("  [license-gated]" if args.require_license else "")
          + ("  [token-auth]" if access_token else "")
          + (f"  [save every {args.save_interval}s]" if args.save_interval else ""))
    try:
        serve(args.path, args.socket, host=args.host, port=args.port,
              license_token=license_token, require_license=args.require_license,
              access_token=access_token, save_interval_s=args.save_interval)
    except KeyboardInterrupt:
        print("\nshutting down")
    except OSError as e:
        print(f"error: cannot bind {where} ({e}); is another telys server already running there?")
        return 2
    except ValueError as e:
        print(f"error: {e}")
        return 2
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="telys", description="Telys — on-device memory & retrieval SDK")
    sub = p.add_subparsers(dest="cmd")
    login = sub.add_parser("login", help="one command: OAuth -> API key -> device -> license -> runtime install")
    login.add_argument("--token", help="use an existing OAuth token (headless/CI); else browser device-code login")
    login.add_argument("--plan", default="telys_developer", help="tier to request (default: telys_developer, free)")
    login.add_argument("--no-install", action="store_true", help="provision only; skip downloading the runtime")
    login.add_argument("--no-browser", action="store_true", help="don't auto-open the browser (print the code)")
    sub.add_parser("version", help="print SDK + format version")
    rt = sub.add_parser("runtime", help="manage the on-device runtime").add_subparsers(dest="rtcmd")
    rt.add_parser("status", help="is the runtime installed?")
    rt_install = rt.add_parser("install", help="install + verify the signed runtime (online host, or --file offline)")
    rt_install.add_argument("--file", help="path to a signed runtime bundle (.bundle/.tar.gz) for offline install")
    rt_install.add_argument("--license", help="path to the signed offline license (license.jwt)")
    rt_install.add_argument("--version", default="latest", help="runtime version to fetch from the host (default: latest)")
    rt.add_parser("verify", help="re-verify the installed runtime's signature/integrity + license (offline)")
    rt.add_parser("update", help="update to the latest runtime on the channel")
    srv = sub.add_parser("serve", help="self-host a shared Telys memory (Team tier) over a Unix socket or TCP")
    srv.add_argument("--path", required=True, help="engine data directory (the shared memory store)")
    srv.add_argument("--socket", help="Unix-domain socket to listen on (trusted host; no token needed)")
    srv.add_argument("--host", help="TCP bind host (e.g. 0.0.0.0 for a container); requires --access-token")
    srv.add_argument("--port", type=int, help="TCP bind port (with --host)")
    srv.add_argument("--access-token", help="shared bearer token for TCP clients; or set TELYS_ACCESS_TOKEN")
    srv.add_argument("--license", help="Telys license token (Team tier); or set TELYS_LICENSE")
    srv.add_argument("--require-license", action="store_true",
                     help="refuse to serve without a valid Telys (products.telys) license")
    srv.add_argument("--save-interval", type=float, default=0.0, metavar="SECONDS",
                     help="background periodic save every N seconds (bounds data loss on a crash; 0=off)")
    srv.add_argument("--supervise", action="store_true",
                     help="run under a supervisor that auto-restarts the server from the last snapshot on crash")

    mcp = sub.add_parser("mcp", help="Telys MCP stdio server (plugin for Claude/Cursor/OpenAI) + one-command install")
    mcp.add_argument("--path", help="memory directory (default: ~/.telys/memory or $TELYS_MEMORY_PATH)")
    mcp.add_argument("--workspace",
                     help=("repo root the auto-indexer maps for telys_repo_search "
                           "(default: $TELYS_MCP_WORKSPACE; unset = pass `path` per call)"))
    mcp.add_argument("--persistent-index", action="store_true",
                     help=("persist the auto-index fingerprint cache under the memory dir so a restart "
                           "skips re-embedding an unchanged repo"))
    mcp.add_argument("--monitor", action="store_true",
                     help=("append every JSON-RPC request/response to <memory dir>/mcp-monitor.jsonl "
                           "so the assistant host's tool usage can be audited"))
    mcp.add_argument("--monitor-log", metavar="PATH",
                     help="monitor log file (implies --monitor; default: <memory dir>/mcp-monitor.jsonl)")
    _mcpsub = mcp.add_subparsers(dest="mcpcmd")  # bare `telys mcp` still runs the server (no subcommand)
    for _verb, _help in (("install", "register the Telys MCP server in an AI client (Claude/Cursor/Codex/Qwen)"),
                         ("uninstall", "deregister the Telys MCP server"),
                         ("status", "show where the Telys MCP server is registered")):
        _sp = _mcpsub.add_parser(_verb, help=_help)
        _sp.add_argument("--name", default="telys", help="server name in the client config")
        if _verb != "status":
            _sp.add_argument("--client", default="all",
                             choices=["claude", "cursor", "codex", "claude-desktop", "qwen", "all"],
                             help="target client (default: auto-detect installed)")
            _sp.add_argument("--scope", default="user", choices=["user", "project"],
                             help="user config (default) or a project-local config file")

    mem = sub.add_parser("mem", help="use Telys memory from the terminal (advanced)").add_subparsers(dest="memcmd")
    for _name in ("add", "search", "create", "list", "stats"):
        mp = mem.add_parser(_name)
        mp.add_argument("--path", help="memory directory (default: ~/.telys/memory)")
        if _name in ("add", "search", "stats"):
            mp.add_argument("-c", "--collection", required=True)
        if _name == "create":
            mp.add_argument("--name", required=True)
            mp.add_argument("--partition-by", default="scope")
        if _name == "add":
            mp.add_argument("-t", "--text", action="append", required=True, help="document text (repeatable)")
            mp.add_argument("--id", action="append", help="document id (repeatable; matches --text order)")
        if _name == "search":
            mp.add_argument("-q", "--query", required=True)
            mp.add_argument("--top-k", type=int, default=5)

    args = p.parse_args(argv)
    if args.cmd == "serve":
        return _serve(args)
    if args.cmd == "version":
        from telys import __version__, FORMAT_VERSION
        print(f"telys {__version__} (format v{FORMAT_VERSION})")
        return 0
    if args.cmd == "login":
        return _login(args)
    if args.cmd == "mcp":
        mcpcmd = getattr(args, "mcpcmd", None)
        if mcpcmd in ("install", "uninstall", "status"):
            from telys import mcp_install
            if mcpcmd == "status":
                return mcp_install.status(name=args.name)
            fn = mcp_install.install if mcpcmd == "install" else mcp_install.uninstall
            return fn(client=args.client, scope=args.scope, name=args.name)
        from telys.mcp import MCPServer, TelysMemory, monitored_stdio
        memory = TelysMemory(getattr(args, "path", None),
                             workspace=getattr(args, "workspace", None),
                             persistent_index=getattr(args, "persistent_index", False))
        log = getattr(args, "monitor_log", None)
        if getattr(args, "monitor", False) and not log:
            log = os.path.join(memory.path, "mcp-monitor.jsonl")
        stdin, stdout = monitored_stdio(log) if log else (None, None)
        return MCPServer(memory).serve(stdin=stdin, stdout=stdout)
    if args.cmd == "mem":
        return _mem(args)
    if args.cmd == "runtime":
        if args.rtcmd == "status":
            return _runtime_status()
        if args.rtcmd == "install":
            return _runtime_install(args)
        if args.rtcmd == "verify":
            return _runtime_verify()
        if args.rtcmd == "update":
            return _not_implemented("runtime update")
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
