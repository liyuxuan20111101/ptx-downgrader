# 真实 llama.cpp kernel 的三重验证(零驱动改动,全程 472.12):
#   1. numpy 参考实现对拍(kernel 有明确数学语义时)
#   2. 参照产物(原生 PTX 仅重写 .target sm_75→sm_61,指令不动)→ ptxas → cubin
#   3. 降级全管线产物 → ptxas → cubin
# 三者在 1070 上执行,参照与降级产物必须逐位一致。
# 当前目标:fwht_cuda_block<1024,256,float>(llama.cpp 的 Walsh-Hadamard 变换)
#
# 关键机制记录(见 README M2.6 节):
#   - 472.12 老 JIT 只接受 .target sm_61(本机架构)的 PTX 模块 → 参照不能
#     直接 JIT,必须走 ptxas 编译为 cubin 再 cuModuleLoadData;
#   - ptxas 的 -arch 必须与 .target 一致,否则 fatal(错误在 stdout);
#   - ctypes 传参:kernel 参数打包进连续缓冲,以指针数组传入。
import ctypes
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "m0"))
sys.path.insert(0, ROOT)

from runner import GPU, check                    # noqa: E402
from runner import cuda as _cuda                 # noqa: E402
from downgrader.parser import parse_module       # noqa: E402
from downgrader.engine import downgrade_module   # noqa: E402

PTXAS = os.path.join(ROOT, "tools", "cuda_nvcc-windows-x86_64-11.4.152-archive",
                     "bin", "ptxas.exe")
DUMP = os.path.join(ROOT, "tools", "full_ptx.txt")
HEAD = ".version 7.0\n.target sm_75\n.address_size 64\n"


def fwht_ref(x, scale):
    """1024 点快速 Walsh-Hadamard 变换 × scale(numpy 参考)。"""
    a = x.copy()
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                u, v = a[j].copy(), a[j + h].copy()
                a[j], a[j + h] = u + v, u - v
        h *= 2
    return a * scale


def find_blob(substr, tag, arch="sm_75"):
    text = open(DUMP, encoding="utf-8", errors="replace").read()
    segs, cur_arch, cur_start = [], None, 0
    for m in re.finditer(r"^arch = (sm_\d+)\s*$", text, flags=re.M):
        if cur_arch:
            segs.append((cur_arch, cur_start, m.start()))
        cur_arch, cur_start = m.group(1), m.end()
    if cur_arch:
        segs.append((cur_arch, cur_start, len(text)))
    for a, s, e in segs:
        if a != arch:
            continue
        seg = text[s:e]
        for m2 in re.finditer(r"\.visible \.entry\s+([\w$]+)\(", seg):
            if substr in m2.group(1) and tag in m2.group(1):
                nxt = re.search(r"\.visible \.entry\s+", seg[m2.end():])
                end = m2.end() + nxt.start() if nxt else len(seg)
                return m2.group(1), seg[m2.start():end]
    return None, None


def to_cubin(ptx_text, tag):
    fd, tp = tempfile.mkstemp(suffix=".ptx")
    with os.fdopen(fd, "w") as f:
        f.write(ptx_text)
    tc = tp + ".cubin"
    r = subprocess.run([PTXAS, "-arch=sm_61", tp, "-o", tc],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{tag}: {r.stdout or r.stderr}")
    os.unlink(tp)
    return tc


def main():
    name, blob = find_blob("fwht_cuda_block", "Li1024ELi256Ef")
    assert name, "未找到 fwht_cuda_block kernel"
    print("kernel:", name[:64])

    gpu = GPU()
    ref_src = (HEAD + blob + "\n").replace(".target sm_75", ".target sm_61")
    cub_ref = to_cubin(ref_src, "ref")
    dg_text, warns = downgrade_module(parse_module(HEAD + blob + "\n"))
    cub_dg = to_cubin(dg_text, "downgraded")
    print("降级改写项:", warns)

    def load(path):
        mod = ctypes.c_void_p()
        check(_cuda.cuModuleLoadData(ctypes.byref(mod), open(path, "rb").read()),
              "load cubin")
        return mod

    mod_ref = load(cub_ref)
    mod_dg = load(cub_dg)

    rng = np.random.default_rng(3)
    N = 1024
    x = rng.standard_normal(N).astype(np.float32)
    scale = np.float32(0.5)
    want = fwht_ref(x, scale)

    d_in = gpu.alloc_h2d(x)
    d_out = gpu.alloc_h2d(np.zeros(N, dtype=np.float32))
    n_blocks = ctypes.c_int64(1)
    scale_arr = np.array([scale], dtype=np.float32)

    def run(mod):
        fn = ctypes.c_void_p()
        check(_cuda.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()),
              "getfn")
        buf = ctypes.create_string_buffer(28)
        ctypes.memmove(buf, ctypes.byref(d_in), 8)
        ctypes.memmove(ctypes.byref(buf, 8), ctypes.byref(d_out), 8)
        ctypes.memmove(ctypes.byref(buf, 16), ctypes.byref(n_blocks), 8)
        ctypes.memmove(ctypes.byref(buf, 24), scale_arr.ctypes.data, 4)
        ptrs = (ctypes.c_void_p * 4)(
            ctypes.cast(buf, ctypes.c_void_p),
            ctypes.cast(ctypes.byref(buf, 8), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(buf, 16), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(buf, 24), ctypes.c_void_p))
        check(_cuda.cuLaunchKernel(fn, 1, 1, 1, 256, 1, 1, 0, None, ptrs, None),
              "launch")
        check(_cuda.cuCtxSynchronize(), "sync")
        got = np.zeros(N, dtype=np.float32)
        check(_cuda.cuMemcpyDtoH_v2(got.ctypes.data_as(ctypes.c_void_p),
                                    d_out, got.nbytes), "dtoh")
        return got

    out_ref = run(mod_ref)
    err_r = float(np.abs(out_ref - want).max())
    print(f"[{'PASS' if err_r < 1e-3 else 'FAIL'}] 参照 vs numpy: {err_r:.2e}")
    check(_cuda.cuMemcpyHtoD_v2(
        d_out, np.zeros(N, dtype=np.float32).ctypes.data_as(ctypes.c_void_p),
        N * 4), "reset")
    out_dg = run(mod_dg)
    err_d = float(np.abs(out_dg - want).max())
    same = np.array_equal(out_ref, out_dg)
    print(f"[{'PASS' if err_d < 1e-3 else 'FAIL'}] 降级 vs numpy: {err_d:.2e}")
    print(f"[{'PASS' if same else 'FAIL'}] 降级 vs 参照逐位一致: {same}")
    os.unlink(cub_ref)
    os.unlink(cub_dg)
    sys.exit(0 if (err_d < 1e-3 and same) else 1)


if __name__ == "__main__":
    main()
