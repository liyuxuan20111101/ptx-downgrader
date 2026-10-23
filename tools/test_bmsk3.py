# Verify bmsk.clamp.b32 3-operand degradation on the real GTX 1070:
# downgrade to sm_61, run the emitted instruction sequence on hardware,
# compare against the official PTX ISA 9.0 §9.7.1.23 semantics in numpy.
import ctypes
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "m0"))
sys.path.insert(0, ROOT)

from runner import GPU, check                            # noqa: E402
from downgrader.parser import parse_module               # noqa: E402
from downgrader.engine import downgrade_module           # noqa: E402
from downgrader.validate import validate                 # noqa: E402

N = 64


def official_bmsk_clamp(a, b):
    """PTX ISA 9.0 §9.7.1.23 .clamp semantics (uint32)."""
    a = int(a) & 0xFFFFFFFF
    b = int(b) & 0xFFFFFFFF
    a1 = a & 0x1F
    mask0 = (~0) << a1
    b1 = b & 0x1F
    total = a1 + b1
    mask1 = (~0) << total
    pos_overflow = a >= 32
    if pos_overflow:
        mask0 = 0                       # documented: mask0 = 0 when a >= 32
    if total >= 32 or pos_overflow or b >= 32:
        mask1 = 0
    elif b1 == 0:
        mask1 = ~0
    return (mask0 & ~mask1) & 0xFFFFFFFF


SRC = """
.version 8.0
.target sm_86
.address_size 64
.visible .entry k(.param .u64 in, .param .u64 inb, .param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [in];
    ld.param.u64 %rd2, [inb];
    ld.param.u64 %rd3, [out];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd5, %r1, 4;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;
    ld.global.b32 %r0, [%rd6];
    ld.global.b32 %r2, [%rd7];
    bmsk.clamp.b32 %r1, %r0, %r2;
    st.global.b32 [%rd8], %r1;
    ret;
}
"""

gpu = GPU()
text, _ = downgrade_module(parse_module(SRC), target="sm_61")
fd, tmp = tempfile.mkstemp(suffix=".ptx", text=True)
with os.fdopen(fd, "w") as f:
    f.write(text)
ok, err = validate(tmp, arch="sm_61")
print("ptxas 11.4 @ sm_61:", "OK" if ok else err[:300])
assert ok
mod = gpu.load_module(tmp)
os.unlink(tmp)

rng = np.random.default_rng(9)
a_in = rng.integers(0, 1 << 32, N, dtype=np.uint32)
b_in = rng.integers(0, 1 << 32, N, dtype=np.uint32)
out = np.zeros(N, dtype=np.uint32)

d0 = gpu.alloc_h2d(a_in)
d1 = gpu.alloc_h2d(b_in)
d2 = gpu.alloc_h2d(out)
fn = gpu.get_function(mod, "k")
gpu.launch(fn, [d0, d1, d2], 1, N)
gpu.dtoh(d2, out)

bad = 0
for i in range(N):
    want = official_bmsk_clamp(a_in[i], b_in[i])
    if out[i] != want:
        bad += 1
        if bad <= 5:
            print(f"  lane {i}: a={a_in[i]:#010x} b={b_in[i]:#010x} "
                  f"got={out[i]:#010x} want={want:#010x}")
print(f"[{'PASS' if bad == 0 else 'FAIL'}] bmsk.clamp 三操作数降级 "
      f"(N={N}): {bad} 不一致")
sys.exit(0 if bad == 0 else 1)
