# 对 vec_add.ptx 里的可疑语法逐个二分
import ctypes

cuda = ctypes.CDLL("nvcuda.dll")
CUresult = ctypes.c_int
for fn, args in [
    ("cuInit", [ctypes.c_int]),
    ("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
    ("cuCtxCreate_v2", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]),
    ("cuModuleLoadDataEx", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                            ctypes.c_uint, ctypes.POINTER(ctypes.c_uint),
                            ctypes.POINTER(ctypes.c_void_p)]),
]:
    f = getattr(cuda, fn)
    f.argtypes = args
    f.restype = CUresult

cuda.cuInit(0)
dev = ctypes.c_int()
cuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)


def try_ptx(name, ptx_src):
    ptx = ptx_src.encode() + b"\0"
    log = ctypes.create_string_buffer(8192)
    keys = (ctypes.c_uint * 2)(3, 4)
    vals = (ctypes.c_void_p * 2)(ctypes.cast(log, ctypes.c_void_p),
                                 ctypes.c_void_p(8192))
    mod = ctypes.c_void_p()
    res = cuda.cuModuleLoadDataEx(ctypes.byref(mod), ptx, 2, keys, vals)
    print(f"{name}: {'OK' if res == 0 else 'FAIL | ' + log.value[:300].decode(errors='replace')}")


HEADER = """
.version 7.0
.target sm_61
.address_size 64
"""

# A: 多行参数列表
try_ptx("A multiline params", HEADER + """
.visible .entry k(
    .param .u64 a,
    .param .u64 b,
    .param .u32 n)
{
    ret;
}
""")

# B: mul.wide.u32 立即数
try_ptx("B mul.wide imm", HEADER + """
.visible .entry k(.param .u64 a)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [a];
    mov.u32 %r1, 3;
    mul.wide.u32 %rd2, %r1, 4;
    add.s64 %rd3, %rd1, %rd2;
    ret;
}
""")

# C: 特殊寄存器参与 add
try_ptx("C add tid.x", HEADER + """
.visible .entry k(.param .u32 n)
{
    .reg .b32 %r<4>;
    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mul.lo.u32 %r2, %r2, %r3;
    add.u32 %r2, %r2, %tid.x;
    ret;
}
""")

# D: 分支 + 标签
try_ptx("D branch label", HEADER + """
.visible .entry k(.param .u32 n)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    ld.param.u32 %r1, [n];
    setp.ge.u32 %p1, %r1, 100;
    @%p1 bra DONE;
    mov.u32 %r2, 0;
DONE:
    ret;
}
""")

# E: 完整复刻 vec_add 主体(带全局内存读写)
try_ptx("E global ld/st", HEADER + """
.visible .entry k(.param .u64 a, .param .u64 c, .param .u32 n)
{
    .reg .f32 %f<4>;
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd3, [c];
    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mul.lo.u32 %r4, %r2, %r3;
    add.u32 %r4, %r4, %tid.x;
    setp.ge.u32 %p1, %r4, %r1;
    @%p1 bra DONE;
    mul.wide.u32 %rd4, %r4, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    st.global.f32 [%rd7], %f1;
DONE:
    ret;
}
""")

# F: vec_add.ptx 原文
try_ptx("F original vec_add", open("ptx/vec_add.ptx").read())
