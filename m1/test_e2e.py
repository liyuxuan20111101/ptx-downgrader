# M1 端到端:新式 PTX → 自动降级 → ptxas 校验 → 1070 实测 vs numpy
# 降级产物只落系统临时目录(tempfile),不写项目目录,用完即删。
import ctypes
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # .../m1
ROOT = os.path.dirname(HERE)                               # 项目根
sys.path.insert(0, os.path.join(ROOT, "m0"))
sys.path.insert(0, ROOT)

from runner import GPU, check                              # noqa: E402
from runner import cuda as _cuda                           # noqa: E402
from downgrader.parser import parse_module                 # noqa: E402
from downgrader.engine import downgrade_module             # noqa: E402
from downgrader.validate import validate                   # noqa: E402


def _downgrade_and_load(gpu, kfile):
    """读新式 kernel → 降级 → tempfile 里过 ptxas → 从临时文件加载模块 → 删。"""
    src = open(os.path.join(HERE, "kernels", kfile), encoding="utf-8").read()
    text, warns = downgrade_module(parse_module(src))
    for w in warns:
        print(f"  [改写] {w}")
    fd, tmp = tempfile.mkstemp(suffix=".ptx", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        ok, err = validate(tmp)
        if not ok:
            raise RuntimeError(f"ptxas 校验失败:\n{err}")
        print("  [ptxas] 校验通过")
        return gpu.load_module(tmp)
    finally:
        os.unlink(tmp)


def run(gpu, mod, entry, args, grid, block):
    fn = gpu.get_function(mod, entry)
    t0 = time.perf_counter()
    gpu.launch(fn, args, grid, block)
    return (time.perf_counter() - t0) * 1e3


def free(gpu, dptr):
    check(_cuda.cuMemFree_v2(dptr), "cuMemFree")


def test_mixed(gpu):
    print("== k13_mixed(特殊寄存器/mov.f16/dp4a.lo/缓存提示) ==")
    mod = _downgrade_and_load(gpu, "k13_mixed.ptx")

    n, block = 1 << 12, 256
    grid = n // block

    # 注意:a16 需 2n 个 fp16 → 视作 n 个 u32(kernel 按 u32 索引),否则越界
    a16 = (np.random.randn(2 * n) * 4).astype(np.float16)
    a8 = np.random.randint(-128, 128, (n, 4), dtype=np.int8)
    out16 = np.zeros(2 * n, dtype=np.float16)   # kernel 写 n 个 u32 = 2n 个 fp16
    out8 = np.zeros(n, dtype=np.int32)

    d0 = gpu.alloc_h2d(a16.view(np.uint32))
    d1 = gpu.alloc_h2d(out16)
    d2 = gpu.alloc_h2d(a8.copy().view(np.int32).ravel())
    d3 = gpu.alloc_h2d(out8)
    dt = run(gpu, mod, "mixed", [d0, d1, d2, d3, ctypes.c_uint32(n)],
             grid, block)
    gpu.dtoh(d1, out16)
    gpu.dtoh(d3, out8)
    for d in (d0, d1, d2, d3):
        free(gpu, d)

    want16 = a16 * np.tile(np.array([2.0, 1.0], dtype=np.float16), n)
    want8 = a8.astype(np.int32).sum(axis=1)
    ok16 = np.allclose(out16, want16, rtol=1e-3, atol=1e-3)
    ok8 = np.array_equal(out8, want8)
    print(f"  [{'PASS' if ok16 else 'FAIL'}] fp16 缩放        {dt:7.2f} ms")
    print(f"  [{'PASS' if ok8 else 'FAIL'}] dp4a int8 求和    {dt:7.2f} ms")
    return ok16 and ok8


def test_async(gpu):
    print("== k13_async(cp.async 16B + src-size 8 零填充 / wait_group) ==")
    mod = _downgrade_and_load(gpu, "k13_async.ptx")

    n_tiles, block = 1024, 64
    grid = n_tiles // block

    # 每 tile 16B:前 8B = 两个 f32(会被拷贝),后 8B 期望零填充;
    # f32[2..3] 不会被读,任意值。
    vals = (np.arange(n_tiles * 2, dtype=np.float32) % 97.5 - 48.0) * 0.25
    src = np.zeros((n_tiles, 4), dtype=np.float32)
    src[:, 0] = vals[0::2]
    src[:, 1] = vals[1::2]
    src = src.ravel()
    out = np.zeros(n_tiles * 4, dtype=np.float32)

    d0 = gpu.alloc_h2d(src)
    d1 = gpu.alloc_h2d(out)
    dt = run(gpu, mod, "async_copy", [d0, d1, ctypes.c_uint32(n_tiles)],
             grid, block)
    gpu.dtoh(d1, out)
    for d in (d0, d1):
        free(gpu, d)

    want = np.zeros((n_tiles, 4), dtype=np.float32)
    want[:, 0] = vals[0::2] * 2.0
    want[:, 1] = vals[1::2] * 2.0
    ok = np.array_equal(out, want.ravel())
    print(f"  [{'PASS' if ok else 'FAIL'}] cp.async 降级+零填充 {dt:7.2f} ms")
    if not ok:
        bad = np.where(out != want.ravel())[0][:6]
        print("  不一致:", bad, out[bad], want.ravel()[bad])
    return ok


def test_mma(gpu):
    print("== k13_mma(mma.sync.m16n8k16 → 共享内存交换 + f32 FMA) ==")
    mod = _downgrade_and_load(gpu, "k13_mma.ptx")

    rng = np.random.default_rng(42)
    A = rng.standard_normal((16, 16)).astype(np.float16)   # 行主序
    Bm = rng.standard_normal((16, 8)).astype(np.float16)   # (k,n) 行主序
    # 交错布局:u32[(n*8)+t] = half2(B[2t][n], B[2t+1][n]),4B 对齐
    B_packed = np.zeros((8, 8), dtype=np.uint32)
    for n in range(8):
        for t in range(8):
            lo = Bm[2 * t, n].view(np.uint16)
            hi = Bm[2 * t + 1, n].view(np.uint16)
            B_packed[n, t] = int(lo) | (int(hi) << 16)
    C = rng.standard_normal((16, 8)).astype(np.float32)
    out = np.zeros(16 * 8, dtype=np.float32)

    d0 = gpu.alloc_h2d(A.reshape(-1).view(np.uint16))
    d1 = gpu.alloc_h2d(B_packed)
    d2 = gpu.alloc_h2d(C)
    d3 = gpu.alloc_h2d(out)
    dt = run(gpu, mod, "mma_test", [d0, d1, d2, d3], 4, 32)
    gpu.dtoh(d3, out)
    for d in (d0, d1, d2, d3):
        free(gpu, d)

    want = (A.astype(np.float32) @ Bm.astype(np.float32) + C).reshape(-1)
    err = float(np.abs(out - want).max())
    ok = np.allclose(out, want, rtol=1e-3, atol=1e-3)
    print(f"  [{'PASS' if ok else 'FAIL'}] mma 展开数值  {dt:7.2f} ms "
          f"(max abs err {err:.2e})")
    if not ok:
        bad = np.argsort(-np.abs(out - want))[:6]
        print("  最大偏差:", bad, out[bad], want[bad])
    return ok


def test_mma8(gpu):
    print("== k13_mma8(m16n8k8 f32.f16.f16.f32 展开) ==")
    src = open(os.path.join(HERE, "kernels", "k13_mma8.ptx"),
               encoding="utf-8").read()
    text, _ = downgrade_module(parse_module(src))
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".ptx", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    from downgrader.validate import validate
    ok_v, err = validate(tmp)
    if not ok_v:
        raise RuntimeError(f"ptxas 校验失败:\n{err}")
    print("  [ptxas] 校验通过")
    mod = gpu.load_module(tmp)
    os.unlink(tmp)

    rng = np.random.default_rng(7)
    A = rng.standard_normal((16, 8)).astype(np.float16)    # (m,k) k=8
    Bm = rng.standard_normal((8, 8)).astype(np.float16)    # (k,n) k=8
    # 交错包:u32[(n*4)+t] = half2(B[2t][n], B[2t+1][n])
    B_packed = np.zeros((8, 4), dtype=np.uint32)
    for n in range(8):
        for t in range(4):
            lo = Bm[2 * t, n].view(np.uint16)
            hi = Bm[2 * t + 1, n].view(np.uint16)
            B_packed[n, t] = int(lo) | (int(hi) << 16)
    C = rng.standard_normal((16, 8)).astype(np.float32)
    out = np.zeros(16 * 8, dtype=np.float32)

    d0 = gpu.alloc_h2d(A.reshape(-1).view(np.uint16))
    d1 = gpu.alloc_h2d(B_packed)
    d2 = gpu.alloc_h2d(C)
    d3 = gpu.alloc_h2d(out)
    dt = run(gpu, mod, "mma8_test", [d0, d1, d2, d3], 1, 32)
    gpu.dtoh(d3, out)
    for d in (d0, d1, d2, d3):
        free(gpu, d)

    want = (A.astype(np.float32) @ Bm.astype(np.float32) + C).reshape(-1)
    err = float(np.abs(out - want).max())
    ok = np.allclose(out, want, rtol=1e-3, atol=1e-3)
    print(f"  [{'PASS' if ok else 'FAIL'}] m16n8k8 展开数值  {dt:7.2f} ms "
          f"(max abs err {err:.2e})")
    if not ok:
        bad = np.argsort(-np.abs(out - want))[:6]
        print("  最大偏差:", bad, out[bad], want[bad])
    return ok


def main():
    gpu = GPU()
    ok = test_mixed(gpu)
    ok &= test_async(gpu)
    ok &= test_mma(gpu)
    ok &= test_mma8(gpu)
    print("\n结论:", "M1 端到端全部通过" if ok else "存在失败项")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
