# Reject-coverage audit: compare the PTX ISA's sm_70+ instruction inventory
# against the downgrader's reject list / supported shapes / special handling,
# to surface instructions that would silently pass through unsupported.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from downgrader.engine import REJECT_PREFIXES          # noqa: E402
from downgrader.mma import SHAPES                      # noqa: E402

known_new = [
    "ldmatrix", "stmatrix", "wmma", "mma.sync", "wgmma", "tcgen05",
    "cp.async", "cp.async.bulk", "tensormap", "mbarrier", "griddepcontrol",
    "clusterlaunchcontrol", "mapa", "barrier.cluster", "setmaxnreg",
    "elect.sync", "redux.sync", "multimem", "st.async", "red.async",
    "cp.reduce.async.bulk", "shfl.sync", "match.sync", "nanosleep",
    "applypriority", "discard", "createpolicy", "st.bulk", "clustergrid",
    "tanh", "bmsk",
]
# Special-cased in the downgrader (not in the reject list but handled):
specially = {"griddepcontrol", "mbarrier", "cp.async", "tanh", "bmsk"}
# Confirmed safe on sm_61: mma.sync (in SHAPES), shfl.sync/match.sync
# (sm_61 has shfl/vote; .sync variants are PTX 6.0 syntax), nanosleep,
# and hint-only instructions (applypriority/discard/createpolicy/clustergrid).
benign = {"mma.sync", "shfl.sync", "match.sync", "nanosleep",
          "applypriority", "discard", "createpolicy", "clustergrid"}
# REJECT_PREFIXES keys are full opcode names (e.g. "cp.async.bulk"), and
# "cp.async" is specially handled — build the covered set from BOTH full
# names and base names so the comparison is apples-to-apples.
covered = (set(REJECT_PREFIXES)
           | {r.split(".")[0] for r in REJECT_PREFIXES}
           | {s.split(".")[0] for s in SHAPES}
           | specially)

missing = [k for k in known_new if k not in covered and k not in benign]
print("uncovered sm_70+ instructions:", missing if missing else "none — full coverage")
