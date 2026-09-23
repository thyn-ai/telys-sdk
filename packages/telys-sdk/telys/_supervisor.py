"""Supervisor for the Telys self-host server — run it as a child process and relaunch on abnormal exit.

The engine runs in-process, so an (already-rare, after the FFI guards in P0/P1) uncatchable native crash takes
the daemon down. The supervisor restarts it from the last on-disk snapshot (collections persist and are lazily
reopened), bounding downtime; periodic save (`telys serve --save-interval`) bounds the data-loss window. A
clean exit (code 0, e.g. a SIGTERM-drained shutdown) is NOT restarted. A crash loop is capped so the
supervisor gives up instead of spinning forever.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time

_log = logging.getLogger("telys.supervisor")


def supervise(child_argv: list[str], *, env: dict | None = None,
              max_restarts: int = 20, window_s: float = 60.0, backoff_s: float = 1.0) -> int:
    """Run `python -m telys.cli <child_argv>` as a child, relaunching on abnormal exit.

    Returns the child's last exit code (0 on a clean drain). Restarts are capped at ``max_restarts`` within
    ``window_s`` to avoid a tight crash loop. SIGTERM/SIGINT are forwarded to the child for a graceful drain.
    """
    cmd = [sys.executable, "-m", "telys.cli", *child_argv]
    child_env = dict(os.environ if env is None else env)
    proc: subprocess.Popen | None = None
    restarts: list[float] = []

    def _forward(signum, _frame):
        if proc is not None and proc.poll() is None:
            proc.send_signal(signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _forward)
        except (ValueError, OSError):
            pass

    while True:
        _log.info("supervisor: starting child: telys %s", " ".join(child_argv))
        proc = subprocess.Popen(cmd, env=child_env)
        try:
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            return proc.wait()
        if rc == 0:
            _log.info("supervisor: child exited cleanly (0) — done")
            return 0
        now = time.monotonic()
        restarts = [t for t in restarts if now - t < window_s]
        restarts.append(now)
        if len(restarts) > max_restarts:
            _log.error("supervisor: child crash-looping (%d restarts in %.0fs) — giving up (rc=%d)",
                       len(restarts), window_s, rc)
            return rc
        _log.warning("supervisor: child died (rc=%d) — restarting from last snapshot in %.1fs", rc, backoff_s)
        time.sleep(backoff_s)
