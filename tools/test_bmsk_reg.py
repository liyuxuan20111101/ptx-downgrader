# Verify bmsk with register-length operand: ptxas + GPU numeric check.
import ctypes
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "m0"))
sys.path.insert(0, os.getcwd())

from runner import GPU                                     # noqa: E402
from downgrader.parser import parse_module                 # noqa: E402
from downgrader.engine import downgrade_module             # noqa: E402
from downgrader.validate import validate                   # noqa: E402

SRC = """
.version 8.0
.target sm_100
.address_size 64
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    mov.u32 %r0, 0xABCD;
    mov.u32 %r2, 7;
    bmsk.clamp.b32 %r1, %r0, %r2, 4;
    st.global.b32 [%rd1], %r1;
    ret;
}
"""
text, _ = downgrade_module(parse_module(SRC), target="sm_86")
fd, tmp = tempfile.mkstemp(suffix=".ptx", text=True)
with os.fdopen(fd, "w") as f:
    f.write(text)
ok, err = validate(tmp)
print("ptxas:", "OK" if ok else err[:300])
assert ok
gpu = GPU()
mod = gpu.load_module(tmp)
os.unlink(tmp)
out = np.zeros(1, dtype=np.uint32)
d0 = gpu.alloc_h2d(out)
fn = gpu.get_function(mod, "k")
gpu.launch(fn, [d0], 1, 1)
gpu.dtoh(d0, out)
print("GPU:", hex(out[0]), "expect 0x3c ->",
      "[PASS]" if out[0] == 0x3C else "[FAIL]")
