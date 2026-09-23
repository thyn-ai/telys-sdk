"""Telys on-disk path conventions for the installed runtime (public SDK, stdlib-only).

Single source of truth for WHERE a verified runtime is installed, shared by:
  - the public installer (`telys.installer`) which writes the verified artifacts here, and
  - the closed runtime's kernel resolvers, which look here to load the installed `.dylib`/`.so`.

No crypto, no engine — just the directory layout. `telys runtime install --file` populates:

    $TELYS_HOME/runtime/<platform>/        (default TELYS_HOME=~/.telys)
        libty_runtime.<ext>                native runtime
        libame_kernel.<ext>                reference kernel
        telys_manifest.json / .sig         the signed manifest that was verified at install
        license.jwt                        the verified offline license
        installed.json                     install record (version, hashes, license summary, verified=true)
"""
from __future__ import annotations

import os
import platform
import sys

# Manifest/install-record artifact names (the basenames the manifest + resolvers agree on).
TY_RUNTIME_STEM = "libty_runtime"
AME_KERNEL_STEM = "libame_kernel"
MANIFEST_NAME = "telys_manifest.json"
MANIFEST_SIG_NAME = "telys_manifest.sig"
LICENSE_NAME = "license.jwt"
INSTALL_RECORD_NAME = "installed.json"
PYSITE_DIR = "pysite"   # verified runtime's Python package (memengine, from the slim telys-runtime-native wheel)


def telys_home() -> str:
    """Root of the Telys on-disk state. Override with $TELYS_HOME (e.g. for tests / multi-tenant installs)."""
    return os.path.abspath(os.environ.get("TELYS_HOME") or os.path.join(os.path.expanduser("~"), ".telys"))


def platform_tag() -> str:
    """Canonical `<os>-<arch>` tag, matching the build-runtime.yml target names (e.g. macos-arm64, linux-x86_64)."""
    m = platform.machine().lower()
    arch = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x86_64", "amd64": "x86_64"}.get(m, m)
    if sys.platform == "darwin":
        osn = "macos"
    elif sys.platform.startswith("linux"):
        osn = "linux"
    elif sys.platform.startswith("win"):
        osn = "windows"
    else:
        osn = sys.platform
    return f"{osn}-{arch}"


def lib_ext() -> str:
    """Shared-library extension for this platform (matches what the build emits)."""
    if sys.platform == "darwin":
        return "dylib"
    if sys.platform.startswith("win"):
        return "dll"
    return "so"


def install_dir(tag: str | None = None) -> str:
    """Directory holding the installed, verified runtime for a platform (default: this platform)."""
    return os.path.join(telys_home(), "runtime", tag or platform_tag())


# Control-plane hosts. `telys login` is the ONLY interactive/networked onboarding step; query execution never
# touches the network. All overridable (env) so the client is testable against a local mock or a pre-cutover host.
DEFAULT_ACCOUNTS_URL = "https://accounts.thyn.ai"   # shared identity portal (Supabase OAuth / device-code)
DEFAULT_API_URL = "https://api.telys.ai"            # control plane (api keys, device register, license); shared backend
DEFAULT_PACKAGES_URL = "https://packages.telys.ai"  # signed-runtime download host


def accounts_url() -> str:
    return (os.environ.get("TELYS_ACCOUNTS_URL") or DEFAULT_ACCOUNTS_URL).rstrip("/")


def api_url() -> str:
    return (os.environ.get("TELYS_API_URL") or DEFAULT_API_URL).rstrip("/")


def packages_url() -> str:
    """Base URL of the runtime-install host. Override with $TELYS_PACKAGES_URL (e.g. a workers.dev URL pre-cutover)."""
    return (os.environ.get("TELYS_PACKAGES_URL") or DEFAULT_PACKAGES_URL).rstrip("/")


def token_path() -> str:
    """Where `telys login` caches the OAuth/session auth token (control-plane auth)."""
    return os.path.join(telys_home(), "token")


def api_key_path() -> str:
    """Where `telys login` caches the created Telys API key (rotatable; control-plane auth for download/renew)."""
    return os.path.join(telys_home(), "api_key")


def device_key_path() -> str:
    """Ed25519 device private key (PEM) binding this machine to its license. Generated once on first login."""
    return os.path.join(telys_home(), "device_key.pem")


def device_id_path() -> str:
    """Stable per-machine device id (opaque), generated once and reused across logins."""
    return os.path.join(telys_home(), "device_id")


def _read_secret(path: str, env: str | None = None) -> str | None:
    if env:
        v = os.environ.get(env)
        if v:
            return v.strip()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return (fh.read().strip() or None)
    return None


def read_token() -> str | None:
    """Resolve the control-plane auth token: $TELYS_TOKEN, else the cached `telys login` token, else the API key."""
    return _read_secret(token_path(), "TELYS_TOKEN") or read_api_key()


def read_api_key() -> str | None:
    """Resolve the Telys API key: $TELYS_API_KEY, else the cached key from `telys login`."""
    return _read_secret(api_key_path(), "TELYS_API_KEY")


def installed_lib_path(stem: str, tag: str | None = None) -> str | None:
    """Absolute path to an installed lib (e.g. installed_lib_path('libty_runtime')) if present, else None.

    Used by the closed kernel resolvers AFTER an explicit env override and BEFORE the in-tree dev path, so a
    `telys runtime install`ed runtime is found with zero configuration."""
    p = os.path.join(install_dir(tag), f"{stem}.{lib_ext()}")
    return p if os.path.exists(p) else None


def installed_pysite(tag: str | None = None) -> str | None:
    """Directory holding the installed runtime's Python package (memengine, from the slim telys-runtime-native
    wheel the signed bundle carries). Added to sys.path after a verified install so the native engine +
    kernel-backed providers import with zero configuration. Returns None if absent."""
    p = os.path.join(install_dir(tag), PYSITE_DIR)
    return p if os.path.isdir(p) else None
