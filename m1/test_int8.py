# GPU verify int8 mma k32 downgrade on GTX 1070.
# Upload uses the SIMULATOR's fragment layout directly (fragments(lane)
# gives the exact per-lane A/B words the expansion expects), eliminating
# upload-layout mismatch as a variable. Then the only thing under test is
# the emitted dp4a sequence itself.
import ctypes
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "m0"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

from runner import GPU, check                            # noqa: E402
from downgrader.parser import parse_module               # noqa: E402
from downgrader.engine import downgrade_module           # noqa: E402
from downgrader.validate import validate                 # noqa: E402
from sim_mma8i import fragments                          # noqa: E402

SRC = """
.version 8.0
.target sm_90
.address_size 64
.visible .entry mma8i(
    .param .u64 up, .param .u64 c, .param .u64 d)
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<24>;
    ld.param.u64    %rd1, [up];
    ld.param.u64    %rd3, [c];
    ld.param.u64    %rd4, [d];
    mov.u32         %r1, %tid.x;
    and.b32         %r2, %r1, 31;
    // 上传 = lane 序 24B 槽:A 4 字 @0/4/8/12,B 2 字 @16/20
    mul.wide.u32    %rd5, %r2, 24;
    add.s64         %rd8, %rd1, %rd5;
    ld.global.b32   %r5, [%rd8+0];
    ld.global.b32   %r6, [%rd8+4];
    ld.global.b32   %r7, [%rd8+8];
    ld.global.b32   %r8, [%rd8+12];
    ld.global.b32   %r9, [%rd8+16];
    ld.global.b32   %r10, [%rd8+20];
    // C(4 x s32):行 g 列 2t/2t+1,行 g+8 —— (16,8) s32 行距 32B
    shr.u32         %r3, %r2, 2;
    and.b32         %r4, %r2, 3;
    mul.wide.u32    %rd17, %r4, 8;
    mul.wide.u32    %rd18, %r3, 32;
    add.s64         %rd19, %rd3, %rd18;
    add.s64         %rd19, %rd19, %rd17;
    ld.global.b32   %r11, [%rd19];
    add.s64         %rd19, %rd19, 4;
    ld.global.b32   %r12, [%rd19];
    add.s64         %rd19, %rd19, 252;
    ld.global.b32   %r13, [%rd19];
    add.s64         %rd19, %rd19, 4;
    ld.global.b32   %r14, [%rd19];

    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 %r11, %r12, %r13, %r14, %r5, %r6, %r7, %r8, %r9, %r10, %r11, %r12, %r13, %r14;

    mul.wide.u32    %rd20, %r3, 32;
    add.s64         %rd21, %rd4, %rd20;
    add.s64         %rd21, %rd21, %rd17;
    st.global.b32   [%rd21], %r11;
    add.s64         %rd21, %rd21, 4;
    st.global.b32   [%rd21], %r12;
    add.s64         %rd21, %rd21, 252;
    st.global.b32   [%rd21], %r13;
    add.s64         %rd21, %rd21, 4;
    st.global.b32   [%rd21], %r14;
    ret;
}
"""


def main():
    gpu = GPU()
    text, _ = downgrade_module(parse_module(SRC), target="sm_61")
    fd, tmp = tempfile.mkstemp(suffix=".ptx", text=True)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    ok, err = validate(tmp, arch="sm_61")
    print("ptxas:", "OK" if ok else err[:300])
    assert ok
    mod = gpu.load_module(tmp)
    os.unlink(tmp)

    rng = np.random.default_rng(17)
    A = rng.integers(-128, 128, (16, 32), dtype=np.int8)
    Bm = rng.integers(-128, 128, (32, 8), dtype=np.int8)
    C = rng.integers(-100, 100, (16, 8), dtype=np.int32)
    out = np.zeros(16 * 8, dtype=np.int32)

    # 上传 = 模拟器 fragment 直出(lane 序 24B 槽)
    up = np.zeros(32 * 6, dtype=np.uint32)
    for lane in range(32):
        fa, fb, fc = fragments(lane)
        for w in range(4):
            bts = [int(A[fa[4 * w + by]]) & 0xFF for by in range(4)]
            up[lane * 6 + w] = (bts[0] | (bts[1] << 8) | (bts[2] << 16)
                                | (bts[3] << 24))
        for w in range(2):
            bts = [int(Bm[fb[4 * w + by][0], fb[4 * w + by][1]]) & 0xFF
                   for by in range(4)]
            up[lane * 6 + 4 + w] = (bts[0] | (bts[1] << 8) | (bts[2] << 16)
                                    | (bts[3] << 24))

    d0 = gpu.alloc_h2d(up)
    d2 = gpu.alloc_h2d(C)
    d3 = gpu.alloc_h2d(out)
    fn = gpu.get_function(mod, "mma8i")
    gpu.launch(fn, [d0, d2, d3], 1, 32)
    gpu.dtoh(d3, out)
    for d in (d0, d2, d3):
        check(__import__("runner").cuda.cuMemFree_v2(d), "free")

    want = (A.astype(np.int32) @ Bm.astype(np.int32) + C).reshape(-1)
    bad = int((out != want).sum())
    print(f"[{'PASS' if bad == 0 else 'FAIL'}] int8 k32 降级(模拟器直出上传): "
          f"{bad} 不一致 / 128")
    if bad:
        idx = np.where(out != want)[0][:4]
        print("  ", idx, out[idx], want[idx])
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
