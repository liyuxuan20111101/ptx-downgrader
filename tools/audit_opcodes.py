# Opcode audit: scan every opcode base name in the sm_75 sections of a
# real fatbin dump and compare against the downgrader's three dispositions
# (supported / rejected / passed through) to surface instructions that
# would leak through unsupported onto sm_61.
# Found this way during development: tanh (3271 uses) and bmsk (1694).
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

DUMP = os.path.join(HERE, "full_ptx.txt")

# Opcodes natively supported on sm_61 (PTX ISA 6.0 and earlier)
SM61_OK = {
    "add", "sub", "mul", "fma", "mad", "mov", "cvt", "ld", "st", "setp",
    "and", "or", "xor", "not", "shl", "shr", "neg", "abs", "min", "max",
    "bar", "bra", "brx", "call", "ret", "exit", "selp", "slct", "rcp",
    "sqrt", "rsqrt", "lg2", "ex2", "sin", "cos", "div", "rem", "sad",
    "dp4a", "dp2a", "shfl", "vote", "bfe", "bfi", "brev", "clz",
    "popc", "prmt", "lop3", "vadd", "vsub", "vabsdiff", "vmnmx", "vset",
    "atom", "red", "membar", "fence", "cvta", "isspacep", "istypep",
    "suld", "sust", "sured", "sred", "trap", "brkpt", "nanosleep",
    "discard", "applypriority", "createpolicy", "getlmid", "match",
    "activemask", "sync", "elect", "redux", "ldu", "copysign",
}
# Handled specially by the downgrader (expanded / stripped / conservatively
# replaced): mma shapes, cp.async, mbarrier, griddepcontrol. tanh and bmsk
# have software degradations since the deep review.
HANDLED = {"mma", "cp", "mbarrier", "griddepcontrol", "tanh", "bmsk"}
# What the downgrader will reject
from downgrader.engine import REJECT_PREFIXES  # noqa: E402

REJECT = set(REJECT_PREFIXES)


def main():
    text = open(DUMP, encoding="utf-8", errors="replace").read()
    segs, cur_arch, cur_start = [], None, 0
    for m in re.finditer(r"^arch = (sm_\d+)\s*$", text, flags=re.M):
        if cur_arch:
            segs.append((cur_arch, cur_start, m.start()))
        cur_arch, cur_start = m.group(1), m.end()
    if cur_arch:
        segs.append((cur_arch, cur_start, len(text)))

    ops = {}
    for arch, s, e in segs:
        if arch != "sm_75":
            continue
        for ln in text[s:e].splitlines():
            s2 = ln.strip()
            if not s2 or s2[0] in ".${}" or s2.startswith("//"):
                continue
            if s2.startswith("@"):
                parts = s2.split(None, 1)
                s2 = parts[1] if len(parts) > 1 else s2
            base = s2.split(None, 1)[0].split(".")[0]
            base = base.rstrip(";")
            if base:
                ops[base] = ops.get(base, 0) + 1

    print(f"sm_75 sections contain {len(ops)} distinct opcode base names")
    leaked = {k: v for k, v in ops.items()
              if k not in SM61_OK and k not in HANDLED and k not in REJECT}
    print("passed through but not on the sm_61 known list (likely leaks):")
    for k, v in sorted(leaked.items(), key=lambda x: -x[1]):
        print(f"  {v:5d} x {k}")
    if not leaked:
        print("  none — full coverage")


if __name__ == "__main__":
    main()
