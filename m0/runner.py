# M0 可行性尖刺:用 Windows 驱动 API(nvcuda.dll)把手写的降级 PTX
# 经驱动 JIT 加载到 sm_61 GPU 上,与 numpy 参考实现比对。
# 零第三方 CUDA 依赖,只需要 numpy。
import ctypes
import os
import time

import numpy as np

cuda = ctypes.CDLL("nvcuda.dll")

CUresult = ctypes.c_int
for fn, args in [
    ("cuInit", [ctypes.c_int]),
    ("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
    ("cuDeviceGetName", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]),
    ("cuCtxCreate_v2", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]),
    ("cuCtxSynchronize", []),
    ("cuModuleLoadDataEx", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                            ctypes.c_uint, ctypes.POINTER(ctypes.c_uint),
                            ctypes.POINTER(ctypes.c_void_p)]),
    ("cuModuleGetFunction", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                             ctypes.c_char_p]),
    ("cuMemAlloc_v2", [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]),
    ("cuMemFree_v2", [ctypes.c_uint64]),
    ("cuMemcpyHtoD_v2", [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]),
    ("cuMemcpyDtoH_v2", [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]),
    ("cuLaunchKernel", [ctypes.c_void_p,
                        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                        ctypes.c_uint, ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]),
    ("cuGetErrorString", [CUresult, ctypes.POINTER(ctypes.c_char_p)]),
]:
    f = getattr(cuda, fn)
    f.argtypes = args
    f.restype = CUresult


def check(res, what):
    if res != 0:
        msg = ctypes.c_char_p()
        cuda.cuGetErrorString(res, ctypes.byref(msg))
        raise RuntimeError(f"{what} failed: CUDA error {res} ({msg.value})")


class GPU:
    def __init__(self):
        check(cuda.cuInit(0), "cuInit")
        self.dev = ctypes.c_int()
        check(cuda.cuDeviceGet(ctypes.byref(self.dev), 0), "cuDeviceGet")
        name = ctypes.create_string_buffer(256)
        check(cuda.cuDeviceGetName(name, 256, self.dev), "cuDeviceGetName")
        self.name = name.value.decode(errors="replace")
        self.ctx = ctypes.c_void_p()
        check(cuda.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, self.dev),
              "cuCtxCreate")
        print(f"device: {self.name}")

    def load_module(self, ptx_path):
        ptx = open(ptx_path, "rb").read() + b"\0"
        # 捕获 JIT 编译日志(CU_JIT_ERROR_LOG_BUFFER=3, SIZE_BYTES=4)
        log = ctypes.create_string_buffer(65536)
        keys = (ctypes.c_uint * 2)(3, 4)
        vals = (ctypes.c_void_p * 2)(ctypes.cast(log, ctypes.c_void_p),
                                     ctypes.c_void_p(65536))
        mod = ctypes.c_void_p()
        res = cuda.cuModuleLoadDataEx(ctypes.byref(mod), ptx, 2, keys, vals)
        if res != 0:
            detail = log.value.decode(errors="replace").strip()
            raise RuntimeError(
                f"cuModuleLoadDataEx({os.path.basename(ptx_path)}) failed: "
                f"CUDA error {res}\n--- JIT log ---\n{detail or '(empty)'}")
        return mod

    def get_function(self, mod, name):
        fn = ctypes.c_void_p()
        check(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()),
              f"cuModuleGetFunction({name})")
        return fn

    def alloc_h2d(self, arr):
        dptr = ctypes.c_uint64()
        check(cuda.cuMemAlloc_v2(ctypes.byref(dptr), arr.nbytes),
              f"cuMemAlloc({arr.nbytes}B)")
        check(cuda.cuMemcpyHtoD_v2(dptr, arr.ctypes.data_as(ctypes.c_void_p),
                                   arr.nbytes), "cuMemcpyHtoD")
        return dptr

    def dtoh(self, dptr, arr):
        check(cuda.cuMemcpyDtoH_v2(arr.ctypes.data_as(ctypes.c_void_p), dptr,
                                   arr.nbytes), "cuMemcpyDtoH")

    def launch(self, fn, args, grid, block):
        bufs = list(args)
        params = (ctypes.c_void_p * len(bufs))(
            *[ctypes.addressof(v) for v in bufs])
        check(cuda.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1,
                                  0, None, params, None), "cuLaunchKernel")
        check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")


def run_kernel(gpu, ptx_name, entry, h_args_in, out_np, grid, block, label):
    mod = gpu.load_module(os.path.join("ptx", ptx_name))
    fn = gpu.get_function(mod, entry)

    d_in = [gpu.alloc_h2d(a) for a in h_args_in]
    d_out = gpu.alloc_h2d(out_np)

    t0 = time.perf_counter()
    gpu.launch(fn, [*d_in, d_out, ctypes.c_uint32(out_np.size)], grid, block)
    dt = (time.perf_counter() - t0) * 1e3

    got = np.empty_like(out_np)
    gpu.dtoh(d_out, got)

    for d in [*d_in, d_out]:
        check(cuda.cuMemFree_v2(d), "cuMemFree")

    return got, dt


def main():
    gpu = GPU()
    n, block = 1 << 20, 256
    grid = (n + block - 1) // block

    # ---- 1. vec_add:最小环路 ----
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    got, dt = run_kernel(gpu, "vec_add.ptx", "vec_add",
                         [a, b], out, grid, block, "vec_add")
    ok = np.allclose(got, a + b, rtol=1e-6, atol=1e-6)
    print(f"[{'PASS' if ok else 'FAIL'}] vec_add    f32  {dt:8.2f} ms")
    if not ok:
        print("  got:", got[:4], " want:", (a + b)[:4])

    # ---- 2. half_scale:fp16 降级路径 ----
    n2 = n // 2  # half2 元素个数
    grid2 = (n2 + block - 1) // block
    a16 = np.random.randn(n).astype(np.float16)
    out16 = np.zeros(n, dtype=np.float16)
    got, dt = run_kernel(gpu, "half_scale.ptx", "half_scale",
                         [a16.view(np.uint32)], out16, grid2, block,
                         "half_scale")
    want = a16 * np.tile(np.array([1.0, 0.5], dtype=np.float16), n2)
    ok = np.allclose(got, want, rtol=1e-3, atol=1e-3)
    print(f"[{'PASS' if ok else 'FAIL'}] half_scale  f16  {dt:8.2f} ms")
    if not ok:
        print("  got:", got[:8], " want:", want[:8])

    # ---- 3. dp4a_dot:int8 降级路径(量化推理基石) ----
    a8 = np.random.randint(-128, 128, (n, 4), dtype=np.int8)
    b8 = np.random.randint(-128, 128, (n, 4), dtype=np.int8)
    out32 = np.zeros(n, dtype=np.int32)
    got, dt = run_kernel(gpu, "dp4a_dot.ptx", "dp4a_dot",
                         [a8.copy().view(np.int32).ravel(),
                          b8.copy().view(np.int32).ravel()],
                         out32, grid, block, "dp4a_dot")
    want = (a8.astype(np.int32) * b8.astype(np.int32)).sum(axis=1)
    ok = np.array_equal(got, want)
    print(f"[{'PASS' if ok else 'FAIL'}] dp4a_dot    i8   {dt:8.2f} ms")
    if not ok:
        bad = np.where(got != want)[0][:4]
        print("  mismatch at", bad, "got", got[bad], "want", want[bad])


if __name__ == "__main__":
    main()
