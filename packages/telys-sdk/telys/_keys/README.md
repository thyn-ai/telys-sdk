# Embedded Telys verification keys (PUBLIC only)

These are the **public** halves of the Telys signing keys, embedded as the default offline trust anchors
(D-31 #3/#5). `telys.verify` loads them when no `TELYS_RELEASE_PUBKEY[_FILE]` / `TELYS_LICENSE_PUBKEY[_FILE]`
override is set, so `telys runtime install --file …` verifies with **zero configuration**.

| File | Verifies | Signed by (private key, NOT in repo) |
|---|---|---|
| `telys_release_pub.pem` | the runtime **manifest** (artifact SHA-256s) | **Telys release key** (Telys-owned, local) |
| `telys_license_pub.pem` | the offline **RS256 license** token (ver:2 entitlement) | the **SHARED platform license key** (decision-engine `license-signer` Worker — the SOLE license key for Telys/Algenta/Codna) |

**Two different authorities (post platform-convergence):**
- **RELEASE** key is Telys-owned and signs runtime manifests (`scripts/telys_issuer.py bundle`). Production-real here.
- **LICENSE** key is NOT Telys-owned — Telys *rides the shared signer*. Real licenses are issued by the
  decision-engine control plane (`api.codna.ai` → the Cloudflare `license-signer` Worker), which alone holds
  the license private key across all products. `telys_license_pub.pem` must therefore be the **decision-engine
  license public key** (export from its JWKS `GET /.well-known/jwks.json` or the operator). `telys.verify`
  checks the engine's ver:2 token (iss `https://license.algenta.ai`, aud `telys-runtime`) against it, offline.

**Override:** `TELYS_LICENSE_PUBKEY[_FILE]`, `TELYS_RELEASE_PUBKEY[_FILE]`, and `TELYS_LICENSE_ISSUER` /
`TELYS_LICENSE_AUDIENCE` env vars override the embedded defaults (rotation / self-hosted / air-gapped).

> ⚠️ **`telys_license_pub.pem` is a BOOTSTRAP platform keypair** (generated locally for dev/staging continuity
> so the offline-license path is exercisable end-to-end before the control plane is live). Its **private half is
> at `dist-keys/platform_license_priv.pem`** (gitignored, issuer custody — NEVER committed). To make real
> licenses verify, do ONE of:
> 1. **Adopt this bootstrap pair (dev/staging):** load `dist-keys/platform_license_priv.pem` into the
>    decision-engine `license-signer` (its `license_signing_private_key` / Cloudflare Worker secret) so the
>    signer mints with the private half of the key embedded here.
> 2. **Production:** generate a **KMS-born** RS256 key in the control plane, then replace `telys_license_pub.pem`
>    with that signer's public half (export from its JWKS `GET /.well-known/jwks.json`) and delete the bootstrap
>    private from `dist-keys/`.
>
> Either way, **one signer authority** holds the license private key across Telys/Algenta/Codna — never mint
> Telys licenses with a separate Telys-local key (the parallel-signer mistake the convergence removed). The
> RELEASE key remains Telys-owned and production-real.
