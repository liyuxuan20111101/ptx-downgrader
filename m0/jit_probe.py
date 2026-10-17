# 探测 472.12 驱动 JIT 接受的 PTX 版本范围
import ctypes

import numpy as np  # noqa: F401  (与 runner 保持同环境)

cuda = ctypes.CDLL("nvcuda.dll")
CUresult = ctypes.c_int
for fn, args in [
    ("cuInit", [ctypes.c_int]),
    ("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
    ("cuCtxCreate_v2", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]),
    ("cuModuleLoadDataEx", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                            ctypes.c_uint, ctypes.POINTER(ctypes.c_uint),
                            ctypes.POINTER(ctypes.c_void_p)]),
    ("cuGetErrorString", [CUresult, ctypes.POINTER(ctypes.c_char_p)]),
]:
    f = getattr(cuda, fn)
    f.argtypes = args
    f.restype = CUresult

cuda.cuInit(0)
dev = ctypes.c_int()
cuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

TEMPLATE = """
.version {ver}
.target sm_61
.address_size 64

.visible .entry noop()
{{
    ret;
}}
"""

for ver in ["4.0", "5.0", "6.0", "6.3", "6.5", "7.0"]:
    ptx = TEMPLATE.format(ver=ver).encode() + b"\0"
    log = ctypes.create_string_buffer(4096)
    keys = (ctypes.c_uint * 2)(3, 4)
    vals = (ctypes.c_void_p * 2)(ctypes.cast(log, ctypes.c_void_p),
                                 ctypes.c_void_p(4096))
    mod = ctypes.c_void_p()
    res = cuda.cuModuleLoadDataEx(ctypes.byref(mod), ptx, 2, keys, vals)
    msg = ctypes.c_char_p()
    cuda.cuGetErrorString(res, ctypes.byref(msg))
    status = "OK" if res == 0 else f"FAIL({msg.value.decode()}) log={log.value[:200]}"
    print(f".version {ver}: {status}")
