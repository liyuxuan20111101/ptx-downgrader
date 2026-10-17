# CI helper: second downgrade of already-downgraded kernels must be a no-op
import pathlib
import sys

sys.path.insert(0, ".")

from downgrader.parser import parse_module
from downgrader.engine import downgrade_module

for p in sorted(pathlib.Path("build").glob("*.sm61.ptx")):
    once, _ = downgrade_module(parse_module(p.read_text(encoding="utf-8")))
    twice, _ = downgrade_module(parse_module(once))
    assert once == twice, f"{p.name}: not idempotent"
    print(f"{p.name}: idempotent OK")
