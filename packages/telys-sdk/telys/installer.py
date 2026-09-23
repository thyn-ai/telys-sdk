"""Telys offline-file runtime installer (public SDK) — `telys runtime install --file <bundle>`.

Proves the universal, platform-neutral distribution model with NO host/auth/CDN (D-31 #3/#4): a signed bundle
+ a signed offline license verify locally, then the runtime runs fully offline. A hosted `packages.telys.ai`
download is a later distribution layer on top of this exact verify path — not a prerequisite.

Bundle = a tar(.gz) containing the signed manifest (telys_manifest.json + .sig) and the artifacts it names
(libty_runtime.<ext>, libame_kernel.<ext>). Install order (fail closed at every step):
  1. read manifest bytes + signature FROM the archive (do not extract yet)
  2. verify the manifest signature over those exact bytes (Telys RELEASE key)
  3. for each manifest artifact: extract ONLY that named member (no path traversal / symlinks), check SHA-256
  4. verify the offline license (strict RS256, Telys LICENSE key) and confirm it grants `products.telys`
  5. atomically place verified files in $TELYS_HOME/runtime/<platform>/ and write installed.json (verified=true)

After install the closed kernel resolvers find the runtime here automatically — zero config, zero network.
"""
from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

from telys import paths
from telys import verify as V


class InstallError(Exception):
    """Install failed (bad bundle, failed verification, or I/O). Nothing is left half-installed."""


# Hard caps to bound memory BEFORE any signature is checked (the bundle is fully attacker-controlled): a tiny
# gzip member can declare a multi-GB size and OOM the host. manifest/sig are tiny by construction; artifacts
# are additionally bounded by the SIGNED manifest's size_bytes when present, else this ceiling.
MAX_MANIFEST_BYTES = 4 * 1024 * 1024          # 4 MB
MAX_SIG_BYTES = 64 * 1024                      # 64 KB
MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024        # 1 GB hard ceiling (per artifact)
MAX_LICENSE_BYTES = 256 * 1024                 # 256 KB


def _safe_member(tar: tarfile.TarFile, name: str) -> tarfile.TarInfo:
    """Fetch a tar member by its BASENAME, rejecting anything unsafe (traversal, symlink, non-file)."""
    if name != os.path.basename(name) or name in ("", ".", ".."):
        raise InstallError(f"unsafe artifact name in manifest: {name!r}")
    try:
        ti = tar.getmember(name)
    except KeyError:
        raise InstallError(f"bundle is missing manifest artifact {name!r}") from None
    if not ti.isfile():
        raise InstallError(f"bundle member {name!r} is not a regular file (symlink/device rejected)")
    if os.path.isabs(ti.name) or ".." in ti.name.split("/"):
        raise InstallError(f"unsafe path in bundle member: {ti.name!r}")
    return ti


def _read_member_bytes(tar: tarfile.TarFile, name: str, max_bytes: int) -> bytes:
    """Read a bundle member with a HARD size cap enforced BEFORE allocation (anti-OOM; pre-verification safe).

    Rejects if the declared TarInfo.size exceeds max_bytes, and reads at most max_bytes+1 so a lying/oversized
    header can't force an unbounded allocation; also rejects a header that lies short (truncated stream)."""
    ti = _safe_member(tar, name)
    if ti.size > max_bytes:
        raise InstallError(f"bundle member {name!r} too large ({ti.size} > {max_bytes} cap)")
    f = tar.extractfile(ti)
    if f is None:
        raise InstallError(f"could not read bundle member {name!r}")
    data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise InstallError(f"bundle member {name!r} exceeds {max_bytes}-byte cap")
    if len(data) != ti.size:
        raise InstallError(f"bundle member {name!r} size mismatch (header {ti.size}, read {len(data)})")
    return data


def _extract_wheel_to_pysite(whl_path: str, pysite_dir: str) -> None:
    """Unzip a VERIFIED runtime wheel (telys-runtime-native) into pysite_dir. A wheel is a zip; extract its
    Python package so the SDK can add pysite to sys.path and import the native engine zero-config. Path-safe
    (reject absolute / traversal members). Called only AFTER the wheel's sha256 matches the signed manifest."""
    os.makedirs(pysite_dir, exist_ok=True)
    with zipfile.ZipFile(whl_path) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            norm = os.path.normpath(member)
            if os.path.isabs(norm) or norm.startswith(".."):
                raise InstallError(f"unsafe path in runtime wheel: {member!r}")
            out = os.path.join(pysite_dir, norm)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with zf.open(member) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)


def install_from_file(bundle_path: str, license_path: str, *, tag: str | None = None) -> dict:
    """Verify a runtime bundle + offline license and install them for this platform. Returns the install record."""
    bundle_path = os.path.abspath(bundle_path)
    if not os.path.exists(bundle_path):
        raise InstallError(f"bundle not found: {bundle_path}")
    if not license_path or not os.path.exists(license_path):
        raise InstallError(f"license file not found: {license_path!r} (pass --license <license.jwt>)")
    tag = tag or paths.platform_tag()

    if os.path.getsize(license_path) > MAX_LICENSE_BYTES:
        raise InstallError(f"license file too large (> {MAX_LICENSE_BYTES} bytes)")
    with open(license_path, "r", encoding="utf-8") as fh:
        token = fh.read().strip()

    # Stage UNDER $TELYS_HOME so the final promote is an atomic same-filesystem rename (no cross-fs copy that
    # could fail mid-way and destroy a good install).
    target = paths.install_dir(tag)
    home_runtime = os.path.dirname(target)
    os.makedirs(home_runtime, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".install-", dir=home_runtime)
    try:
        with tarfile.open(bundle_path, "r:*") as tar:
            # 1+2: manifest bytes + signature (HARD-capped reads), verified before anything is trusted/written.
            manifest_bytes = _read_member_bytes(tar, paths.MANIFEST_NAME, MAX_MANIFEST_BYTES)
            sig = _read_member_bytes(tar, paths.MANIFEST_SIG_NAME, MAX_SIG_BYTES)
            manifest = V.verify_manifest(manifest_bytes, sig)        # raises on bad signature
            if manifest.get("platform") not in (None, tag):
                raise InstallError(
                    f"bundle platform {manifest.get('platform')!r} != this platform {tag!r}"
                )
            # 3: extract + integrity-check each named artifact (only members the SIGNED manifest lists).
            #    Bound each read by the signed size_bytes (trusted, post-signature) capped at MAX_ARTIFACT_BYTES.
            verified_files = []
            pysite_installed = False
            for a in manifest["artifacts"]:
                name = a["name"]
                want = a.get("size_bytes")
                cap = min(int(want), MAX_ARTIFACT_BYTES) if isinstance(want, int) and want >= 0 else MAX_ARTIFACT_BYTES
                data = _read_member_bytes(tar, name, cap)
                if isinstance(want, int) and len(data) != want:
                    raise InstallError(f"artifact {name!r} size mismatch vs signed manifest")
                dest = os.path.join(staging, name)
                with open(dest, "wb") as out:
                    out.write(data)
                V.verify_artifact(dest, a["sha256"])                 # raises on hash mismatch
                verified_files.append((name, dest, a["sha256"]))
                if name.endswith(".whl"):
                    # The signed slim runtime wheel (telys-runtime-native): extract its Python package into a
                    # telys-managed pysite dir so the SDK imports the native engine zero-config. Drop the raw
                    # wheel afterward — only its extracted contents are installed.
                    _extract_wheel_to_pysite(dest, os.path.join(staging, paths.PYSITE_DIR))
                    os.remove(dest)
                    pysite_installed = True

        # 4: license — strict offline RS256, must grant products.telys.
        claims = V.verify_license(token, now=int(time.time()))

        # 5: the verified artifacts already sit in `staging` under their real names. Add the metadata files,
        #    then atomically promote staging -> target on the SAME filesystem.
        with open(os.path.join(staging, paths.MANIFEST_NAME), "wb") as fh:
            fh.write(manifest_bytes)
        with open(os.path.join(staging, paths.MANIFEST_SIG_NAME), "wb") as fh:
            fh.write(sig)
        with open(os.path.join(staging, paths.LICENSE_NAME), "w", encoding="utf-8") as fh:
            fh.write(token)

        record = {
            "verified": True,
            "platform": tag,
            "telys_format_version": __import__("telys").FORMAT_VERSION,
            "installed_at": int(time.time()),
            "source_bundle": os.path.basename(bundle_path),
            "pysite": pysite_installed,   # the runtime's Python package (memengine) is installed under pysite/
            "artifacts": [{"name": n, "sha256": s} for n, _, s in verified_files],
            "license": {
                "iss": claims.get("iss"), "exp": claims.get("exp", claims.get("expires_at")),
                "tier": (claims.get("products", {}).get(V.PRODUCT, {}) or {}).get("tier"),
                "features": (claims.get("products", {}).get(V.PRODUCT, {}) or {}).get("features"),
                "license_id": claims.get("license_id"),
            },
        }
        with open(os.path.join(staging, paths.INSTALL_RECORD_NAME), "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)

        # Atomic same-filesystem promote with rollback: move any existing good install aside, rename the new
        # one into place, then drop the backup. If the final rename fails, restore the previous install.
        backup = target + f".old-{os.getpid()}"
        had_old = os.path.exists(target)
        if had_old:
            os.replace(target, backup)                  # atomic rename away (same fs)
        try:
            os.replace(staging, target)                 # atomic rename into place (same fs)
        except OSError:
            if had_old:
                os.replace(backup, target)              # roll back to the previous good install
            raise
        staging = None  # promoted; don't clean it
        if had_old:
            shutil.rmtree(backup, ignore_errors=True)
        return record
    finally:
        if staging and os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)


MAX_BUNDLE_DOWNLOAD = 4 * 1024 * 1024 * 1024   # 4 GB ceiling on a downloaded bundle (defense-in-depth)


def _http_download(url: str, token: str | None, dest_path: str, max_bytes: int) -> str:
    """Stream a URL to dest_path with a Bearer token + a hard size cap. The ONLY network call in the SDK."""
    headers = {"User-Agent": "telys-installer"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as out:
            total = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise InstallError(f"download exceeds {max_bytes}-byte cap: {url}")
                out.write(chunk)
    except urllib.error.HTTPError as e:
        hint = " (set TELYS_TOKEN or run `telys login`)" if e.code in (401, 403) else ""
        raise InstallError(f"download failed [{e.code}] {url}: {e.reason}{hint}") from e
    except urllib.error.URLError as e:
        raise InstallError(f"could not reach install host {url}: {e.reason}") from e
    return dest_path


# Platforms the closed on-device runtime is built + published for (the Mojo targets: Modular ships Mojo only
# for these). Everything else gets a clear message, never a raw 404 — Windows runs under WSL2 (the Linux build),
# and Intel Mac isn't a Mojo target yet.
SUPPORTED_PLATFORMS = ("macos-arm64", "linux-x86_64", "linux-arm64")

# Experimental (P1) targets: built best-effort in CI (the leg is `experimental: true`, so a transient toolchain
# failure can leave a given release without this bundle). Installable when a bundle IS published; a missing bundle
# is an honest "not published for this release yet" note for a best-effort target — NOT a broken-install signal.
EXPERIMENTAL_PLATFORMS = ("linux-arm64",)


def _unsupported_platform_note(tag: str) -> str | None:
    """A human message if this platform can't run the Telys runtime, else None."""
    if tag in SUPPORTED_PLATFORMS:
        return None
    supported = ", ".join(SUPPORTED_PLATFORMS)
    if tag.startswith("windows"):
        return (
            "The Telys on-device runtime isn't built for Windows natively — run it under WSL2 (Ubuntu), where "
            f"`telys` uses the Linux build. Supported platforms: {supported}."
        )
    if tag == "macos-x86_64":
        return (
            "The Telys on-device runtime isn't available for Intel Macs yet (it currently targets Apple "
            f"Silicon). Supported platforms: {supported}."
        )
    return f"No Telys runtime is published for this platform ({tag}) yet. Supported platforms: {supported}."


def install_from_host(*, version: str = "latest", base_url: str | None = None, token: str | None = None,
                      license_path: str | None = None, tag: str | None = None) -> dict:
    """Online install (D-31 #4): download the signed bundle (+ license) from the host, then verify LOCALLY.

    Download is the ONLY networked step; verification + execution are offline (reuses install_from_file, so the
    same RS256/SHA-256 gate applies). Resolves host via base_url|$TELYS_PACKAGES_URL|packages.telys.ai and the
    token via token|$TELYS_TOKEN|`telys login` cache. License resolution order: --license, then the cached
    `telys login` license ($TELYS_HOME/login_license.jwt), then GET {host}/license (OEM download-token hosts).
    """
    base_url = (base_url or paths.packages_url()).rstrip("/")
    token = token or paths.read_token()
    tag = tag or paths.platform_tag()
    note = _unsupported_platform_note(tag)
    if note:
        raise InstallError(note)  # don't bother the network for a platform we don't ship
    staging = tempfile.mkdtemp(prefix="telys-dl-")
    try:
        bundle_url = f"{base_url}/runtime/{version}/{tag}.bundle"
        try:
            bundle = _http_download(bundle_url, token, os.path.join(staging, "rt.bundle"), MAX_BUNDLE_DOWNLOAD)
        except InstallError as exc:
            if "[404]" in str(exc):  # supported platform, but no bundle uploaded for this version yet
                if tag in EXPERIMENTAL_PLATFORMS:
                    stable = ", ".join(p for p in SUPPORTED_PLATFORMS if p not in EXPERIMENTAL_PLATFORMS)
                    raise InstallError(
                        f"The Telys runtime for {tag} is an experimental (best-effort) target and no bundle is "
                        f"published at version '{version}' yet. Fully-supported platforms: {stable}. "
                        f"Try `--version latest`, or point TELYS_PACKAGES_URL at a host that carries it."
                    ) from exc
                raise InstallError(
                    f"No Telys runtime bundle is published for {tag} at version '{version}' yet "
                    f"(supported: {', '.join(SUPPORTED_PLATFORMS)}). Try `--version latest` or check back soon."
                ) from exc
            raise
        if license_path:
            lic = license_path
        else:
            # Self-service reuse: `telys login` caches the control-plane license here — reinstall/upgrade must
            # NOT fall through to GET {host}/license (that route serves OEM download-token licenses only, so a
            # self-service user 404s there). OEM hosts keep working: no cached license -> GET /license.
            cached = os.path.join(paths.telys_home(), "login_license.jwt")
            if os.path.exists(cached):
                lic = cached
            else:
                lic = _http_download(f"{base_url}/license", token, os.path.join(staging, "license.jwt"), MAX_LICENSE_BYTES)
        rec = install_from_file(bundle, lic, tag=tag)
        rec["source"] = bundle_url
        return rec
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def install_record(tag: str | None = None) -> dict | None:
    """Read the install record for a platform, or None if nothing is installed."""
    p = os.path.join(paths.install_dir(tag), paths.INSTALL_RECORD_NAME)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_installed(tag: str | None = None) -> dict:
    """Re-verify an installed runtime end-to-end (manifest signature, artifact hashes, license). Raises on failure.

    Returns a small report. This is what `telys runtime verify` runs — fully offline, no mutation."""
    tag = tag or paths.platform_tag()
    d = paths.install_dir(tag)
    if not os.path.exists(os.path.join(d, paths.INSTALL_RECORD_NAME)):
        raise InstallError(f"no runtime installed for {tag} (run `telys runtime install --file <bundle>`)")

    def _read_capped(name, cap):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            raise InstallError(f"installed file missing: {name}")
        if os.path.getsize(p) > cap:
            raise InstallError(f"installed {name} too large (> {cap} bytes)")
        with open(p, "rb") as fh:
            return fh.read()

    manifest_bytes = _read_capped(paths.MANIFEST_NAME, MAX_MANIFEST_BYTES)
    sig = _read_capped(paths.MANIFEST_SIG_NAME, MAX_SIG_BYTES)
    manifest = V.verify_manifest(manifest_bytes, sig)
    checked = []
    for a in manifest["artifacts"]:
        name = a["name"]
        if name.endswith(".whl"):
            # The slim runtime wheel is consumed into pysite/ at install time and its raw file is intentionally
            # removed (see install_from_file). It is still listed in the signed manifest, so re-verify the
            # extracted payload is present rather than re-hashing a wheel that no longer exists on disk.
            if not os.path.isdir(os.path.join(d, paths.PYSITE_DIR)):
                raise InstallError(f"installed pysite/ missing (expected from {name})")
            checked.append(f"{name} -> {paths.PYSITE_DIR}/")
            continue
        path = os.path.join(d, name)
        if not os.path.exists(path):
            raise InstallError(f"installed artifact missing: {name}")
        V.verify_artifact(path, a["sha256"])
        checked.append(name)
    with open(os.path.join(d, paths.LICENSE_NAME), "r", encoding="utf-8") as fh:
        token = fh.read().strip()
    claims = V.verify_license(token, now=int(time.time()))
    return {
        "ok": True, "platform": tag, "dir": d, "artifacts_verified": checked,
        "tier": (claims.get("products", {}).get(V.PRODUCT, {}) or {}).get("tier"),
        "exp": claims.get("exp", claims.get("expires_at")),
    }
