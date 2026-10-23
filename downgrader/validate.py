# ptxas 本地校验:降级产物的第一道关卡(比驱动 JIT 的报错精确到行)。
# ptxas 版本按目标架构选择:11.4 的 ptxas 只认到 sm_86 之前的 target
# (实测报 "PTX .version 7.0 does not support .target sm_86"),目标为
# sm_80/86/89 时必须用 CUDA 13 的 ptxas。
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")

PTXAS_114 = os.path.join(TOOLS, "cuda_nvcc-windows-x86_64-11.4.152-archive",
                         "bin", "ptxas.exe")
PTXAS_130 = os.path.join(TOOLS, "cuda_nvcc-windows-x86_64-13.0.88-archive",
                         "bin", "ptxas.exe")

# 各目标对应的 ptxas 路径
BY_ARCH = {
    "sm_61": PTXAS_114,
    "sm_70": PTXAS_114,
    "sm_75": PTXAS_114,
    "sm_80": PTXAS_130,
    "sm_86": PTXAS_130,
    "sm_89": PTXAS_130,
}


def validate(ptx_path, arch="sm_61"):
    ptxas = BY_ARCH.get(arch, PTXAS_114)
    if not os.path.isfile(ptxas):
        ptxas = PTXAS_114            # 回退(老目标不受影响)
    r = subprocess.run([ptxas, f"-arch={arch}", ptx_path, "-o", os.devnull],
                       capture_output=True, text=True)
    return r.returncode == 0, ((r.stderr or "") + (r.stdout or "")).strip()
