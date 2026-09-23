"""telys login — one-command onboarding for external users (D-32).

`telys login` authenticates via the shared accounts portal (OAuth 2.0 device grant, RFC 8628), creates a free
Telys Developer API key, registers this device, receives a platform-signed offline license, and installs +
verifies the signed on-device runtime. After that, Telys runs fully offline (no query-time network).

Two secrets, distinct roles (deliberate):
  - API key   — rotatable; control-plane auth for runtime download / renewal / device registration.
  - license   — RS256, product-scoped (products.telys) + device-scoped; lets the installed runtime run offline.

Hosts are env-overridable (TELYS_ACCOUNTS_URL / TELYS_API_URL / TELYS_PACKAGES_URL) so the whole flow is
testable against a local mock. Stdlib urllib only (no new SDK deps); cryptography (already a telys dep) for the
device keypair.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import time
import urllib.error
import urllib.request
import uuid

from telys import __version__ as _sdk_version
from telys import paths

_USER_AGENT = f"telys-cli/{_sdk_version} python-urllib/{platform.python_version()}"

_MAX_RESP_BYTES = 1 << 20  # control-plane JSON is tiny; cap defensively


class LoginError(RuntimeError):
    """A step of the login/onboarding flow failed."""


# ── device identity: a stable id + an Ed25519 keypair whose thumbprint binds the license to this machine ──────

def device_id() -> str:
    p = paths.device_id_path()
    existing = paths._read_secret(p)  # noqa: SLF001 — same-package helper
    if existing:
        return existing
    value = uuid.uuid4().hex
    _cache(p, value)
    return value


def _load_or_make_device_key():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    p = paths.device_key_path()
    if os.path.exists(p):
        with open(p, "rb") as fh:
            return serialization.load_pem_private_key(fh.read(), password=None)
    key = ed25519.Ed25519PrivateKey.generate()
    os.makedirs(paths.telys_home(), exist_ok=True)
    pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    with open(p, "wb") as fh:
        fh.write(pem)
    os.chmod(p, 0o600)
    return key


def device_thumbprint() -> str:
    """`ed25519:<b64url(sha256(raw_pub))>` — matches the control plane's device-binding format."""
    from cryptography.hazmat.primitives import serialization

    raw = _load_or_make_device_key().public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return "ed25519:" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()


def _hostname_hash() -> str:
    """Stable, privacy-preserving machine fingerprint. The control plane requires it for finite-device plans
    (e.g. telys_developer) to enforce the device limit / detect the same physical machine re-registering; a
    missing value is rejected with `device_fingerprint_required`."""
    import socket

    return hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()


def _cache(path: str, value: str) -> None:
    os.makedirs(paths.telys_home(), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(value.strip() + "\n")
    os.chmod(path, 0o600)


# ── HTTP (stdlib; small JSON; Bearer) ────────────────────────────────────────────────────────────────────────

def _request(method: str, url: str, *, bearer: str | None = None, body: dict | None = None,
             timeout: float = 30.0) -> tuple[int, dict]:
    # UA is mandatory: the api.telys.ai zone WAF rejects the urllib default
    # ("Python-urllib/X.Y") with Cloudflare Error 1010 browser_signature_banned.
    # Setting a stable, identifiable UA also gives the edge a way to allowlist
    # legitimate CLI traffic separately from anonymous urllib bot noise.
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Accept": "application/json", "User-Agent": _USER_AGENT}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(_MAX_RESP_BYTES + 1)
            status = resp.status
    except urllib.error.HTTPError as e:  # 4xx/5xx — capture the body (device-flow signals errors here)
        raw = b""
        try:
            raw = e.read(_MAX_RESP_BYTES + 1)
        except Exception:  # noqa: BLE001
            pass
        status = e.code
    except urllib.error.URLError as e:
        raise LoginError(f"{method} {url}: {e.reason}") from e
    if len(raw) > _MAX_RESP_BYTES:
        raise LoginError(f"{url}: response too large")
    try:
        payload = json.loads(raw or b"{}")
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return status, payload


# ── OAuth 2.0 device authorization grant (RFC 8628) ──────────────────────────────────────────────────────────

def device_authorize(*, accounts: str, open_browser: bool = True, max_wait: float = 300.0) -> str:
    _, start = _request("POST", f"{accounts}/oauth/device/code",
                        body={"client_id": "telys-cli", "scope": "telys"})
    device_code = start.get("device_code")
    user_code = start.get("user_code")
    verify = start.get("verification_uri_complete") or start.get("verification_uri")
    interval = float(start.get("interval") or 5)
    if not (device_code and user_code and verify):
        raise LoginError("accounts device-code response missing fields")
    print(f"\n  Open {verify}\n  and enter code:  {user_code}\n")
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(verify)
        except Exception:  # noqa: BLE001 — headless boxes have no browser; the code is printed above
            pass
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        time.sleep(interval)
        status, tok = _request("POST", f"{accounts}/oauth/device/token",
                               body={"client_id": "telys-cli", "device_code": device_code,
                                     "grant_type": "urn:ietf:params:oauth:grant-type:device_code"})
        access = tok.get("access_token")
        if access:
            return access
        error = tok.get("error")
        if error in ("authorization_pending", "slow_down") or status in (400, 428):
            continue
        raise LoginError(f"device authorization failed: {error or f'HTTP {status}'}")
    raise LoginError("timed out waiting for device authorization")


# ── control-plane calls (match decision-engine: POST /v1/telys/onboard, POST /v1/device/register) ────────────

def create_api_key(*, api: str, access_token: str, plan: str) -> str:
    # Supabase-JWT entry point: /v1/telys/onboard resolves-or-creates the org on the free telys_developer tier
    # (so the device license carries products.telys) AND mints the first API key. (POST /v1/api-keys does NOT
    # accept a Supabase JWT — it requires an existing API key/license — which is why onboarding has its own route.)
    status, resp = _request("POST", f"{api}/v1/telys/onboard", bearer=access_token,
                            body={"name": "telys-cli", "product": "telys", "plan": plan})
    key = resp.get("raw_key") or resp.get("key") or resp.get("api_key")
    if status >= 400 or not key:
        raise LoginError(f"could not create API key (HTTP {status})")
    return key


def register_device(*, api: str, api_key: str) -> str:
    # The control plane's RegisterRequest nests the device identity under `device` (DeviceInfo) and, for
    # finite-device plans, requires a hostname_hash fingerprint. A flat body 422s ("body.device required");
    # omitting the fingerprint 400s ("device_fingerprint_required"). device_id (uuid4 hex, 32 chars) meets the
    # server's min_length=16. Verified end-to-end against api.telys.ai.
    status, resp = _request("POST", f"{api}/v1/device/register", bearer=api_key,
                            body={"device": {"device_id": device_id(),
                                             "device_pubkey_thumbprint": device_thumbprint(),
                                             "platform": paths.platform_tag(),
                                             "hostname_hash": _hostname_hash()}})
    license_token = resp.get("license_token") or resp.get("license")
    if status >= 400 or not license_token:
        # Surface the server's structured error code so users can act on it:
        # local_runtime_not_included / device_platform_required / device_fingerprint_required /
        # device_key_mismatch — see decision-engine apps/api_server/routers/devices.py:164-208 + :299-312.
        err = resp.get("error") if isinstance(resp, dict) else None
        detail = (err or {}).get("message") if isinstance(err, dict) else None
        code = (err or {}).get("code") if isinstance(err, dict) else None
        suffix = ""
        if code:
            suffix += f" [{code}]"
        if detail:
            suffix += f": {detail}"
        raise LoginError(f"device registration failed (HTTP {status}){suffix}")
    return license_token


# ── orchestration: the whole `telys login` in one call ───────────────────────────────────────────────────────

def login(*, plan: str = "telys_developer", access_token: str | None = None, install: bool = True,
          open_browser: bool = True) -> dict:
    accounts, api = paths.accounts_url(), paths.api_url()
    token = access_token or os.environ.get("TELYS_TOKEN")
    if token:
        print("using supplied token (headless)")
    else:
        print(f"authenticating via {accounts} …")
        token = device_authorize(accounts=accounts, open_browser=open_browser)
    _cache(paths.token_path(), token)

    print("creating your Telys Developer API key …")
    # Clear any prior-session credentials up-front so a mid-flow failure never leaves the CLI pointing at a
    # stale (revoked / wrong-org) key + license. Without this, a failed create_api_key raise leaves
    # ~/.telys/api_key at its previous value, which then 401s against packages.telys.ai on the next
    # `telys runtime install` — indistinguishable from a real backend outage.
    for stale in (paths.api_key_path(), os.path.join(paths.telys_home(), "login_license.jwt")):
        try:
            os.unlink(stale)
        except FileNotFoundError:
            pass
        except OSError:
            pass  # best-effort; a locked / permission-denied file is caller's problem
    api_key = create_api_key(api=api, access_token=token, plan=plan)
    _cache(paths.api_key_path(), api_key)

    print("registering this device …")
    license_token = register_device(api=api, api_key=api_key)
    lic_path = os.path.join(paths.telys_home(), "login_license.jwt")
    _cache(lic_path, license_token)

    installed = None
    if install:
        print("installing + verifying the on-device runtime …")
        from telys.installer import install_from_host

        installed = install_from_host(token=api_key, license_path=lic_path)

    tier = (installed or {}).get("license", {}).get("tier") if installed else plan
    print("\n  ✓ telys login complete — Telys now runs fully offline (no network at query time).\n")
    return {"device_id": device_id(), "installed": bool(installed), "tier": tier,
            "api_key_prefix": (api_key.split("_", 1)[0] + "_…") if "_" in api_key else api_key[:6] + "…"}
