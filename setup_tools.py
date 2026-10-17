# One-time setup: download the small NVIDIA tool binaries this project needs
# (cuobjdump + ptxas, ~10 MB each) from NVIDIA's official redistribution
# servers. Nothing here bypasses any EULA: these components are freely
# redistributable per NVIDIA's CUDA redist policy, and this script fetches
# them exactly like the official CUDA installer does.
#
# Usage:  python setup_tools.py
import json
import socket
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

# All filesystem access goes through pathlib paths rooted at this script's
# directory; every fetched name is validated against strict patterns before
# use, and zip members are checked for traversal before extraction.
HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
TOOLS.mkdir(exist_ok=True)

# Hard allow-list: only NVIDIA's official redistribution host, HTTPS only.
ALLOWED_HOST = "developer.download.nvidia.com"
BASE = f"https://{ALLOWED_HOST}/compute/cuda/redist/"

# (manifest, component, needed_for)
WANT = [
    ("redistrib_11.4.4.json", "cuda_nvcc",
     "ptxas - local PTX validation (v11.4 matches driver 472.x / PTX ISA 7.0)"),
    ("redistrib_11.4.4.json", "cuda_cuobjdump",
     "cuobjdump v11.4 - legacy fatbins (optional)"),
    ("redistrib_13.0.1.json", "cuda_cuobjdump",
     "cuobjdump v13 - CUDA 13.x fatbins (required for survey)"),
]


class _SSRFGuard(urllib.request.BaseHandler):
    """Block redirects and DNS rebinding: every response host and every
    resolved IP must be the allow-listed NVIDIA host on a public address."""

    def __init__(self, host):
        self.host = host

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError(f"redirect blocked (allow-list is single-host): {newurl}")

    def http_open(self, req):
        host = urllib.parse.urlsplit(req.full_url).hostname
        if host != self.host:
            raise RuntimeError(f"blocked non-allow-listed host: {host}")
        for info in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP):
            ip = info[4][0]
            parts = [int(x) for x in ip.split(".")]
            private = (parts[0] == 10 or parts[0] == 127
                       or (parts[0] == 172 and 16 <= parts[1] <= 31)
                       or (parts[0] == 192 and parts[1] == 168)
                       or (parts[0] == 169 and parts[1] == 254))
            if private:
                raise RuntimeError(f"blocked private address: {ip}")
        return self.parent.open(req, timeout=120)

    https_open = http_open


def tools_path(name):
    """Return TOOLS/name after strict validation; refuses any traversal.
    Zip members may be nested ('archive/LICENSE') or directories
    (trailing '/'); the resolve()-based containment check below is the
    authoritative traversal guard."""
    clean = name.rstrip("/")
    if (not clean or Path(clean).is_absolute() or ".." in Path(clean).parts
            or clean.startswith("/")):
        raise RuntimeError(f"unsafe name: {name!r}")
    p = (TOOLS / clean).resolve()
    if p != TOOLS.resolve() and TOOLS.resolve() not in p.parents:
        raise RuntimeError(f"escapes tools dir: {name!r}")
    return p


def fetch_manifest(name):
    path = tools_path(name)
    if path.is_file():
        return json.loads(path.read_text())
    opener = urllib.request.build_opener(_SSRFGuard(ALLOWED_HOST))
    url = BASE + name
    print(f"fetch {url}")
    data = opener.open(url).read()
    path.write_bytes(data)
    return json.loads(data)


def fetch_component(manifest, comp):
    if comp not in manifest or "windows-x86_64" not in manifest[comp]:
        raise RuntimeError(f"component not in manifest: {comp}")
    rel = manifest[comp]["windows-x86_64"]["relative_path"]
    if (".." in rel or not rel.startswith(comp + "/windows-x86_64/")
            or not rel.endswith(".zip")):
        raise RuntimeError(f"unexpected relative_path: {rel}")
    zipname = Path(rel).name
    zpath = tools_path(zipname)
    dest = tools_path(zipname[:-4])
    if dest.is_dir():
        print(f"  {comp}: already present, skipping")
        return
    opener = urllib.request.build_opener(_SSRFGuard(ALLOWED_HOST))
    url = BASE + rel
    print(f"fetch {url}")
    resp = opener.open(url)
    total = int(resp.headers.get("Content-Length", 0))
    done = 0
    with open(zpath, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r  {zipname}: {done/1e6:6.1f}/{total/1e6:.1f} MB")
                sys.stdout.flush()
    print()
    with zipfile.ZipFile(zpath) as z:
        for member in z.namelist():
            tools_path(member)                 # zip-slip guard
        for member in z.namelist():
            z.extract(member, TOOLS)
    zpath.unlink()
    print(f"  extracted {comp}")


def main():
    for manifest_name, comp, why in WANT:
        m = fetch_manifest(manifest_name)
        print(f"[{comp}] {why}")
        fetch_component(m, comp)
    print("\nDone. Verify with:")
    print("  tools/cuda_nvcc-windows-x86_64-11.4.152-archive/bin/ptxas.exe --version")
    print("  tools/cuda_cuobjdump-windows-x86_64-13.0.85-archive/bin/cuobjdump.exe --version")


if __name__ == "__main__":
    main()
