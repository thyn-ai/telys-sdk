"""Telys offline verification (public SDK): runtime manifest signature, artifact integrity, RS256 license.

This is the trust core of the on-device model (D-31 #3/#5): the runtime + license verify themselves locally,
offline, against Telys public keys — no network, no OS-native signing. Mirrors the existing Algenta stack
(RS256 license verified against an operator-supplied public key) and codna's canonical-JSON integrity model.

Three separate trust anchors (NEVER share private keys across them — D-31 key separation):
  - RELEASE key  → signs the runtime manifest (artifact SHA-256s).   verify_manifest()
  - LICENSE key  → signs the RS256 entitlement token.                verify_license()
  - DEVICE key   → Ed25519, per-machine binding (optional; not handled here).

Public keys resolve in order: env inline PEM → env *_FILE → embedded telys/_keys/<name>.pem. Production embeds
the Telys public keys; the env/file override stays for rotation / air-gapped operators (as in decision-engine).

CRITICAL canonicalization rule (D-31 #5 / your guidance): the manifest signature covers the EXACT manifest
bytes shipped. We verify those bytes as-read and NEVER re-serialize/re-canonicalize before verifying.

Requires `cryptography` (a core SDK dep); imported lazily with a clear message so `import telys` stays light.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os

# Telys rides the SHARED platform license signer (the decision-engine `license-signer` Cloudflare Worker —
# one private key for all products). Licenses are the engine's ver:2 entitlement tokens; we verify them
# OFFLINE against the engine's RS256 license public key (embedded telys/_keys/telys_license_pub.pem, or env
# override). iss/aud are env-overridable for self-hosted/air-gapped operators.
DEFAULT_LICENSE_ISSUER = "https://license.algenta.ai"   # the shared signer's issuer
# The shared license-signer Worker tags tokens with ONE platform-runtime audience (`algenta-runtime`); accept
# it (product scoping is via products.telys), plus a telys-specific aud in case the worker is later split.
DEFAULT_LICENSE_AUDIENCES = ("telys-runtime", "algenta-runtime")
SUPPORTED_LICENSE_VER = 2                               # the platform's entitlement claim version
PRODUCT = "telys"


def _expected_issuer() -> str:
    return os.environ.get("TELYS_LICENSE_ISSUER") or DEFAULT_LICENSE_ISSUER


def _expected_audiences() -> tuple:
    env = os.environ.get("TELYS_LICENSE_AUDIENCE")   # comma-separated override
    return tuple(a.strip() for a in env.split(",") if a.strip()) if env else DEFAULT_LICENSE_AUDIENCES


class VerificationError(Exception):
    """Any signature / integrity / license-claim check failed. Fail closed — never proceed on this."""


def _crypto():
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        return InvalidSignature, hashes, serialization, padding
    except ImportError as e:  # noqa: BLE001
        raise VerificationError(
            "offline verification needs the `cryptography` package — `pip install telys` pulls it in; "
            "if you vendored the SDK, `pip install cryptography>=42`."
        ) from e


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _load_public_key(env_inline: str, env_file: str, embedded_name: str):
    """Resolve a verification public key PEM (inline env > file env > embedded) and parse it. Fail if none."""
    _, _, serialization, _ = _crypto()
    pem = os.environ.get(env_inline)
    if not pem:
        path = os.environ.get(env_file)
        if not path:
            here = os.path.dirname(__file__)
            cand = os.path.join(here, "_keys", embedded_name)
            path = cand if os.path.exists(cand) else None
        if path:
            with open(path, "rb") as fh:
                pem = fh.read().decode("utf-8")
    if not pem:
        raise VerificationError(
            f"no Telys verification key found. Set {env_inline} (inline PEM) or {env_file} (PEM path), "
            f"or ship the embedded key telys/_keys/{embedded_name}. (Production embeds it; the env override "
            "is for key rotation / air-gapped operators.)"
        )
    try:
        key = serialization.load_pem_public_key(pem.encode("utf-8") if isinstance(pem, str) else pem)
    except Exception as e:  # noqa: BLE001
        raise VerificationError(f"invalid public key PEM ({embedded_name}): {e}") from e
    # Pin the trust-anchor key TYPE to RSA — we only ever verify RS256 (no EC/Ed25519 key-type confusion).
    from cryptography.hazmat.primitives.asymmetric import rsa
    if not isinstance(key, rsa.RSAPublicKey):
        raise VerificationError(f"Telys verification key must be RSA, got {type(key).__name__} ({embedded_name})")
    return key


def release_public_key():
    return _load_public_key("TELYS_RELEASE_PUBKEY", "TELYS_RELEASE_PUBKEY_FILE", "telys_release_pub.pem")


def license_public_key():
    return _load_public_key("TELYS_LICENSE_PUBKEY", "TELYS_LICENSE_PUBKEY_FILE", "telys_license_pub.pem")


def _rsa_verify(pubkey, signature: bytes, message: bytes) -> None:
    """RSASSA-PKCS1-v1_5 + SHA-256 over `message`. Raises VerificationError on any failure."""
    InvalidSignature, hashes, _, padding = _crypto()
    try:
        pubkey.verify(signature, message, padding.PKCS1v15(), hashes.SHA256())
    except InvalidSignature as e:
        raise VerificationError("signature does not verify against the Telys public key") from e


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(manifest_bytes: bytes, signature: bytes, pubkey=None) -> dict:
    """Verify the manifest SIGNATURE over the EXACT shipped bytes, then parse + sanity-check it.

    Returns the parsed manifest dict {format_version, platform, artifacts:[{name,sha256,size_bytes}]}.
    We verify the raw bytes first and only THEN json.loads — never the other way around (no re-canonicalization)."""
    if pubkey is None:
        pubkey = release_public_key()
    _rsa_verify(pubkey, signature, manifest_bytes)          # signature covers the exact bytes as shipped
    try:
        manifest = json.loads(manifest_bytes)
    except Exception as e:  # noqa: BLE001
        raise VerificationError(f"manifest verified but is not valid JSON: {e}") from e
    if not isinstance(manifest, dict) or not isinstance(manifest.get("artifacts"), list) or not manifest["artifacts"]:
        raise VerificationError("manifest missing a non-empty 'artifacts' list")
    for a in manifest["artifacts"]:
        if not (isinstance(a, dict) and isinstance(a.get("name"), str) and isinstance(a.get("sha256"), str)
                and len(a["sha256"]) == 64):
            raise VerificationError(f"manifest artifact entry malformed: {a!r}")
    return manifest


def verify_artifact(path: str, expected_sha256: str) -> None:
    """Confirm a file's SHA-256 matches the (already signature-verified) manifest entry."""
    actual = sha256_file(path)
    if not _consttime_eq(actual, expected_sha256.lower()):
        raise VerificationError(
            f"artifact integrity check FAILED for {os.path.basename(path)}: "
            f"expected {expected_sha256[:12]}…, got {actual[:12]}…"
        )


def _consttime_eq(a: str, b: str) -> bool:
    import hmac as _hmac
    return _hmac.compare_digest(a, b)


def verify_license(token: str, pubkey=None, *, now: int, issuer: str = None, audience=None,
                   product: str = PRODUCT) -> dict:
    """STRICT offline RS256 verification of the shared platform's ver:2 entitlement license. Returns claims.

    Enforced: alg==RS256 ONLY (no 'none', no HS*, no alg negotiation from the header); reject any JWS `crit`
    header; signature must verify against the platform LICENSE key; iss must match the shared signer; aud must
    include the Telys audience; ver==2; expiry REQUIRED (exp), checked against `now` (+ offline_grace_days);
    the product entry (products.<product>) must exist with a non-empty tier AND non-empty features list.
    """
    if issuer is None:
        issuer = _expected_issuer()
    accepted_aud = set(_expected_audiences()) if audience is None else (
        {audience} if isinstance(audience, str) else set(audience))
    if pubkey is None:
        pubkey = license_public_key()
    parts = token.strip().split(".")
    if len(parts) != 3:
        raise VerificationError("license is not a compact JWS (expected header.payload.signature)")
    h_b64, p_b64, s_b64 = parts
    try:
        header = json.loads(_b64url_decode(h_b64))
    except Exception as e:  # noqa: BLE001
        raise VerificationError(f"license header is not valid base64url JSON: {e}") from e
    # ---- header policy: pin the algorithm; refuse negotiation / critical extensions ----
    if header.get("alg") != "RS256":
        raise VerificationError(f"license alg must be RS256, got {header.get('alg')!r} (no 'none'/HS*/negotiation)")
    if "crit" in header:
        raise VerificationError("license declares unsupported critical ('crit') header extensions — rejected")
    # ---- signature over the EXACT signing input bytes ----
    signing_input = (h_b64 + "." + p_b64).encode("ascii")
    _rsa_verify(pubkey, _b64url_decode(s_b64), signing_input)
    try:
        claims = json.loads(_b64url_decode(p_b64))
    except Exception as e:  # noqa: BLE001
        raise VerificationError(f"license payload is not valid base64url JSON: {e}") from e
    # ---- claim policy (treat bools as invalid for numeric/version fields; reject non-finite numbers) ----
    import math
    if not isinstance(claims, dict):
        raise VerificationError("license payload is not a JSON object")
    if claims.get("iss") != issuer:
        raise VerificationError(f"license issuer {claims.get('iss')!r} != expected {issuer!r}")
    aud = claims.get("aud")                          # JWT aud may be a string or a list; require an accepted one
    aud_toks = {aud} if isinstance(aud, str) else (set(aud) if isinstance(aud, list) else set())
    if not (aud_toks & accepted_aud):
        raise VerificationError(f"license audience {aud!r} not in accepted {sorted(accepted_aud)}")
    ver = claims.get("ver")
    if isinstance(ver, bool) or not isinstance(ver, int) or ver != SUPPORTED_LICENSE_VER:
        raise VerificationError(f"license ver {ver!r} unsupported (this client supports v{SUPPORTED_LICENSE_VER})")
    exp = claims.get("exp", claims.get("expires_at"))
    if isinstance(exp, bool) or not isinstance(exp, (int, float)) or not math.isfinite(exp):
        raise VerificationError("license has no finite numeric expiry ('exp') — expiry is required")
    graceval = claims.get("offline_grace_days", claims.get("grace_days", 0))
    if isinstance(graceval, bool) or not isinstance(graceval, (int, float)) or not math.isfinite(graceval) or graceval < 0:
        raise VerificationError(f"license has an invalid offline_grace_days: {graceval!r}")
    if now > int(exp) + int(graceval) * 86400:
        raise VerificationError(f"license expired (exp={int(exp)}, grace_days={int(graceval)}, now={now})")
    products = claims.get("products")
    if not isinstance(products, dict) or product not in products:
        raise VerificationError(f"license does not grant product {product!r}")
    entry = products[product]
    if not isinstance(entry, dict) or not entry.get("tier"):
        raise VerificationError(f"license product {product!r} missing a tier")
    feats = entry.get("features")
    if not isinstance(feats, list) or not feats:
        raise VerificationError(f"license product {product!r} missing a non-empty features list")
    return claims
