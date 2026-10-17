# ptxas 本地校验:降级产物的第一道关卡(比驱动 JIT 的报错精确到行)
import subprocess
import sys
import os

PTXAS = os.path.join(os.path.dirname(__file__), "..", "tools",
                     "cuda_nvcc-windows-x86_64-11.4.152-archive", "bin",
                     "ptxas.exe")


def validate(ptx_path, arch="sm_61"):
    r = subprocess.run([PTXAS, f"-arch={arch}", ptx_path, "-o", os.devnull],
                       capture_output=True, text=True)
    return r.returncode == 0, ((r.stderr or "") + (r.stdout or "")).strip()
