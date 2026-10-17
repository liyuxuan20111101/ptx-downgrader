# Compatibility survey: extract every kernel's PTX from a CUDA 13 fatbin
# and run it through the downgrader, classifying each kernel as pass/reject
# with a rejection-reason histogram.
#
# Usage: python survey.py <ggml-cuda.dll | cuobjdump-exported full_ptx.txt>
import os
import re
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from downgrader.parser import parse_module, UnsupportedError   # noqa: E402
from downgrader.engine import downgrade_module                 # noqa: E402

CUOBJDUMP = os.path.join(HERE, "cuda_cuobjdump-windows-x86_64-13.0.85-archive",
                         "bin", "cuobjdump.exe")


def _split_blobs(lines):
    """Split the cuobjdump dump on 'arch = sm_XX' header lines."""
    blobs = []
    cur_arch, cur_lines = None, []
    for ln in lines:
        m = re.match(r"\s*arch\s*=\s*(sm_\d+)", ln)
        if m:
            if cur_arch and cur_lines:
                blobs.append((cur_arch, "\n".join(cur_lines)))
            cur_arch, cur_lines = m.group(1), []
            continue
        if cur_arch is not None:
            cur_lines.append(ln)
    if cur_arch and cur_lines:
        blobs.append((cur_arch, "\n".join(cur_lines)))
    return blobs


def extract_ptx_from_text(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    return _split_blobs(text.splitlines())


def extract_ptx(dll):
    # Only accept existing .dll/.so files; subprocess.run with an argument
    # list (no shell), tolerating stray binary bytes in the output.
    dll = os.path.realpath(os.path.abspath(dll))
    if not (dll.endswith(".dll") or dll.endswith(".so")) or not os.path.isfile(dll):
        raise SystemExit(f"input must be an existing .dll/.so file: {dll}")
    completed = subprocess.run([CUOBJDUMP, "-ptx", dll], shell=False,
                               capture_output=True,
                               errors="replace")
    if completed.returncode != 0 and not completed.stdout:
        raise RuntimeError(completed.stderr[:500])
    return _split_blobs(completed.stdout.splitlines())


def split_modules(text):
    """cuobjdump concatenates .entry blocks; split on .visible .entry."""
    parts = re.split(r"(?=\.visible \.entry|\.\s*visible\s+\.entry)", text)
    return [p for p in parts if ".entry" in p]


def survey(blobs):
    stats = {}
    for arch, text in blobs:
        mods = split_modules(text)
        s = stats.setdefault(arch, {"kernels": 0, "ok": 0, "reject": 0,
                                    "reject_reasons": Counter(),
                                    "reject_ops": Counter()})
        s["kernels"] += len(mods)
        for mtext in mods:
            try:
                mod = parse_module(mtext)
                if not mod["entries"]:
                    s["kernels"] -= 1
                    continue
                downgrade_module(mod)
                s["ok"] += 1
            except UnsupportedError as e:
                s["reject"] += 1
                msg = str(e)
                om = re.search(r"`([^`]+)`", msg)
                if om:
                    op = om.group(1).split(".")[0]
                    s["reject_ops"][op] += 1
                rm = re.search(r"— (.+)$", msg)
                reason = rm.group(1) if rm else msg[:60]
                s["reject_reasons"][reason] += 1
        stats[arch] = s

    print()
    for arch, s in sorted(stats.items()):
        print(f"== {arch}: {s['kernels']} kernels, "
              f"pass {s['ok']}, reject {s['reject']}")
        for reason, n in s["reject_reasons"].most_common(8):
            print(f"   {n:4d} x {reason}")
        if s["reject_ops"]:
            print(f"   ops: {dict(s['reject_ops'].most_common(10))}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python survey.py <ggml-cuda.dll | full_ptx.txt>")
    arg = sys.argv[1]
    if arg.endswith(".txt"):
        survey(extract_ptx_from_text(arg))
    else:
        survey(extract_ptx(arg))
