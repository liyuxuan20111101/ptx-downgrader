# CI helper: ptxas-validate every downgraded kernel
import pathlib
import sys

sys.path.insert(0, ".")

from downgrader.validate import validate

ok_all = True
for p in sorted(pathlib.Path("build").glob("*.sm61.ptx")):
    ok, err = validate(str(p))
    if not ok:
        print(f"{p.name}: FAILED\n{err}")
        ok_all = False
    else:
        print(f"{p.name}: ptxas OK")
sys.exit(0 if ok_all else 1)
