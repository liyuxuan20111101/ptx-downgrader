# CI helper: downgrade every test kernel into build/
import pathlib
import sys

sys.path.insert(0, ".")

from downgrader.parser import parse_module
from downgrader.engine import downgrade_module

out_dir = pathlib.Path("build")
out_dir.mkdir(exist_ok=True)
for p in sorted(pathlib.Path("m1/kernels").glob("*.ptx")):
    text, warns = downgrade_module(parse_module(p.read_text(encoding="utf-8")))
    out = out_dir / (p.stem + ".sm61.ptx")
    out.write_text(text, encoding="utf-8")
    print(f"{p.name}: downgraded, {len(text.splitlines())} lines")
