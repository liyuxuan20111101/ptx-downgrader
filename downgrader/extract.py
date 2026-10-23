# fatbin / 二进制内的 PTX 段提取:被 CLI survey 与 GUI 共用。
# 只做子进程调用 + 文本切分,不加载任何外部输入为代码。
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CUOBJDUMP = os.path.join(HERE, "..", "tools",
                         "cuda_cuobjdump-windows-x86_64-13.0.85-archive",
                         "bin", "cuobjdump.exe")


def _split_blobs(lines):
    """Split the cuobjdump dump on 'arch = sm_XX' header lines."""
    blobs = []
    cur_arch, cur_lines = None, []
    for ln in lines:
        m = re.match(r"\s*arch\s*=\s*(sm_\d+)", ln)
        if m:
            if cur_arch and cur_lines:
                blobs.append((cur_arch, "\n".join(cur_lines)))
            cur_arch, cur_lines = m.group(1), []
            continue
        if cur_arch is not None:
            cur_lines.append(ln)
    if cur_arch and cur_lines:
        blobs.append((cur_arch, "\n".join(cur_lines)))
    return blobs


def extract_ptx(path):
    """从 .dll/.so/.cubin/.fatbin 提取 (arch, ptx_text) 列表。

    返回 [] 表示文件里没有嵌入 PTX 段(如纯 SASS 构建),
    抛异常表示 cuobjdump 无法处理该文件。
    """
    real = os.path.realpath(os.path.abspath(path))
    ext = os.path.splitext(real)[1].lower()
    if ext in (".dll", ".so", ".cubin", ".fatbin"):
        if not os.path.isfile(real):
            raise SystemExit(f"文件不存在: {real}")
    else:
        raise SystemExit(f"不支持的输入类型: {real} (需要 .dll/.so/.cubin/.fatbin)")
    completed = subprocess.run([CUOBJDUMP, "-ptx", real], shell=False,
                               capture_output=True, errors="replace")
    if completed.returncode != 0 and not completed.stdout:
        raise RuntimeError(completed.stderr[:500])
    return _split_blobs(completed.stdout.splitlines())


def extract_ptx_from_text(path):
    """从 cuobjdump 导出的文本读 PTX(测试/离线用)。"""
    text = open(path, encoding="utf-8", errors="replace").read()
    return _split_blobs(text.splitlines())


def split_modules(text):
    """cuobjdump concatenates .entry blocks; split on .visible .entry."""
    parts = re.split(r"(?=\.visible \.entry|\.\s*visible\s+\.entry)", text)
    return [p for p in parts if ".entry" in p]
