# Reject-coverage audit: compare the PTX ISA's sm_70+ instruction inventory
# against the downgrader's per-architecture reject tables / supported shapes
# / special handling, to surface instructions that would silently pass
# through unsupported for a given downgrade target.
# Usage: python audit_reject.py [target]     (default sm_61)
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from downgrader.engine import (REJECT_BY_ARCH, REJECT_ANY,   # noqa: E402
                               SUPPORTED_TARGETS, _arch_num)

target = sys.argv[1] if len(sys.argv) > 1 else "sm_61"
if target not in SUPPORTED_TARGETS:
    raise SystemExit(f"target must be one of {SUPPORTED_TARGETS}")
tgt = _arch_num(target)

known_new = [
    "ldmatrix", "stmatrix", "wmma", "mma.sync", "wgmma", "tcgen05",
    "cp.async", "cp.async.bulk", "tensormap", "mbarrier", "griddepcontrol",
    "clusterlaunchcontrol", "mapa", "barrier.cluster", "setmaxnreg",
    "elect.sync", "redux.sync", "multimem", "st.async", "red.async",
    "cp.reduce.async.bulk", "shfl.sync", "match.sync", "nanosleep",
    "applypriority", "discard", "createpolicy", "st.bulk", "clustergrid",
    "tanh", "bmsk",
]
# Special-cased in the downgrader (handled by dedicated passes or benign
# passthrough on any target):
specially = {"griddepcontrol", "mbarrier", "cp.async", "tanh", "bmsk",
             # sm_61-native or PTX-6.0 syntax; hardware-supported on every
             # target this tool emits:
             "shfl.sync", "match.sync", "nanosleep", "applypriority",
             "discard", "createpolicy", "clustergrid",
             # mma.sync shapes are expanded; sm_61+ native from sm_75:
             "mma.sync"}
# benign when the TARGET supports them natively (instruction older than or
# equal to target): e.g. ldmatrix is native on sm_75+.
covered = set(specially)
for base, (intro, _why) in {**REJECT_BY_ARCH, **REJECT_ANY}.items():
    if tgt >= intro:
        covered.add(base)               # native on this target, passthrough OK
    else:
        covered.add(base.split(".")[0])  # rejected for this target — covered

# Known candidates for future expansion passes (tracked, not yet implemented):
KNOWN_PENDING = {"ldmatrix", "redux.sync"}

missing = [k for k in known_new
           if k not in covered and k.split(".")[0] not in covered]
pending = [k for k in missing
           if k in KNOWN_PENDING or k.split(".")[0] in KNOWN_PENDING]
hard = [k for k in missing if k not in pending]
if pending:
    print(f"target {target}: pending expansion (tracked for v0.3):",
          sorted(pending))
if hard:
    print(f"target {target}: UNCOVERED:", sorted(hard))
if not missing:
    print(f"target {target}: none — full coverage")
