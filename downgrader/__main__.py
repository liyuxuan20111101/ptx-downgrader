# CLI: python -m downgrader input.ptx [-o output.ptx]
import argparse
import os
import sys

# Windows consoles default to legacy codepages (cp1252/cp936) that cannot
# encode the CJK status output below; force UTF-8 everywhere.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

from .parser import parse_module, UnsupportedError
from .engine import downgrade_module
from .validate import validate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _safe_out(p):
    """输出路径规范化并限制在项目目录内。"""
    p = os.path.realpath(os.path.abspath(p))
    if p != ROOT and not p.startswith(ROOT + os.sep):
        sys.exit(f"输出路径必须在项目目录内: {p}")
    return p


def main():
    ap = argparse.ArgumentParser(
        description="PTX 降级器:把高代架构 PTX 重写为低代目标可加载的形式")
    ap.add_argument("input", nargs="?", default=None)
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--gui", action="store_true",
                    help="启动图形界面(免命令行,开箱即用)")
    ap.add_argument("--target", default="sm_61",
                    help="降级目标架构:sm_61 (Pascal) / sm_70 / sm_75 / "
                         "sm_80 / sm_86 (Ampere, 30系) / sm_89 (Ada)")
    args = ap.parse_args()

    if args.gui or args.input is None:
        from .gui import main as gui_main
        return gui_main()

    src = open(args.input, encoding="utf-8").read()
    try:
        mod = parse_module(src)
        out_text, warns = downgrade_module(mod, target=args.target)
    except UnsupportedError as e:
        print(f"[不支持的指令] {e}", file=sys.stderr)
        sys.exit(2)

    suffix = "sm" + args.target[3:]
    out_path = _safe_out(args.output or (args.input.rsplit(".", 1)[0]
                                         + f".{suffix}.ptx"))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_text)
    for w in warns:
        print(f"[改写] {w}")

    if not args.no_validate:
        ok, err = validate(out_path, arch=args.target)
        if ok:
            print(f"[ptxas] {out_path}: 校验通过")
        else:
            print(f"[ptxas] 校验失败:\n{err}", file=sys.stderr)
            sys.exit(3)
    print(f"[完成] {out_path}")


if __name__ == "__main__":
    main()
