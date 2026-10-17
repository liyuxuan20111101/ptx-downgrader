# mma 的软件展开。
# 原理:张量核的 D=A·B+C 里每个 D 元素是完整点积,但 A/B 数据分散在同一
# warp 的不同 lane 手里(硬件内部做交换)。软件展开用共享内存做这个交换:
# 各 lane 把自己的 fragment 存进共享内存,bar.sync 后每个 lane 从别的 lane
# 读全 A 行 / B 列,用 f32 FMA 算完整点积。
# k16 布局与展开公式已由 tools/sim_mma.py 对拍 numpy 验证;
# k8 布局已由 tools/sim_mma8.py 对拍 numpy 验证(布局公式均取自 PTX ISA 9.0)。
from .parser import Inst, UnsupportedError

# 形状 → (k 长度, 输入类型, 累加器类型)
SHAPES = {
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32": (16, "f16", "f32"),
    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16": (16, "f16", "f16"),
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32": (16, "bf16", "f32"),
    "mma.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32": (16, "tf32", "f32"),
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32": (8, "f16", "f32"),
    "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16": (8, "f16", "f16"),
}
SHARED_BYTES_PER_WARP = 24 * 32          # lane 存 4 个 A 字 + 2 个 B 字
WARPS = 32                               # 静态共享缓冲按 32 warp 预留
SHARED_TOTAL = SHARED_BYTES_PER_WARP * WARPS
SHARED_VAR = "__dg_mma"

DECLS = [
    ".reg .b32 %dgmb<48>;",
    ".reg .b64 %dgmd<12>;",
    ".reg .f16 %dgmh<4>;",
    ".reg .b16 %dgmw<4>;",
    ".reg .f32 %dgmf<28>;",
]


def _gen(inst, k_len, itype, atype):
    ops = inst.operands
    k8 = (k_len == 8)
    # 操作数:f32 acc = D4 + A4 + B(2 k16 / 1 k8) + C4;f16 acc 的 D/C 各 2 字
    if atype == "f32":
        n_expect = 11 if k8 else 14
        if len(ops) != n_expect:
            raise UnsupportedError(
                f"行{inst.line_no}: mma 操作数数量 {len(ops)} != {n_expect}")
        if k8:
            d, a, b, c = ops[0:4], ops[4:6], ops[6:7], ops[7:11]
        else:
            d, a, b, c = ops[0:4], ops[4:8], ops[8:10], ops[10:14]
    else:
        n_expect = 9 if k8 else 10
        if len(ops) != n_expect:
            raise UnsupportedError(
                f"行{inst.line_no}: mma(f16 acc) 操作数数量 {len(ops)} != {n_expect}")
        if k8:
            d, a, b, c = ops[0:2], ops[2:6], ops[6:7], ops[7:9]
        else:
            d, a, b, c = ops[0:2], ops[2:6], ops[6:8], ops[8:10]
    W = [f"%dgmb{i}" for i in range(48)]
    D = [f"%dgmd{i}" for i in range(12)]
    # 注意池名不能是 W 池名的前缀+数字(%dgmb16 与 %dgmb<48> 的成员
    # %dgmb16 撞名会让 ptxas 报 Illegal expression),用独立名 %dgmw
    B16 = [f"%dgmw{i}" for i in range(4)]
    H = ["%dgmh0", "%dgmh1"]
    F = [f"%dgmf{i}" for i in range(28)]
    L = []
    tf32 = (itype == "tf32")

    def emit(op, *operands):
        L.append(Inst(inst.pred, op, list(operands), inst.line_no))

    def unpack(word, fa, fb):
        """一个 f16x2/bf16x2 字 → 两个 f32。tf32 字直接单转。"""
        if tf32:
            emit("cvt.f32.tf32", fa, word)
            return
        emit("mov.b32", "{" + H[0] + ", " + H[1] + "}", word)
        cvt = "cvt.f32.f16" if itype == "f16" else "cvt.f32.bf16"
        emit(cvt, fa, H[0])
        emit(cvt, fb, H[1])

    # ---- 线性 tid / lane / warp / g / t ----
    emit("mov.u32", W[0], "%tid.x")
    emit("mov.u32", W[1], "%ntid.x")
    emit("mov.u32", W[2], "%tid.y")
    emit("mov.u32", W[3], "%ntid.y")
    emit("mov.u32", W[4], "%tid.z")
    emit("mul.lo.u32", W[5], W[2], W[1])
    emit("mul.lo.u32", W[6], W[1], W[3])
    emit("mul.lo.u32", W[7], W[6], W[4])
    emit("add.u32", W[8], W[0], W[5])
    emit("add.u32", W[8], W[8], W[7])             # 线性 tid
    emit("and.b32", W[9], W[8], "31")             # lane
    emit("shr.u32", W[10], W[8], "5")             # warp(块内)
    emit("shr.u32", W[11], W[9], "2")             # g
    emit("and.b32", W[12], W[9], "3")             # t

    # ---- 共享内存地址:基址 + warp*768 + lane*24 ----
    emit("mov.u64", D[0], SHARED_VAR)
    emit("mul.wide.u32", D[1], W[10], str(SHARED_BYTES_PER_WARP))
    emit("add.s64", D[0], D[0], D[1])
    emit("mul.wide.u32", D[2], W[9], "24")
    emit("add.s64", D[3], D[0], D[2])             # 本 lane 存储地址

    # ---- 存 fragment(两种 k 的 A 都占 4 字、B 占 2 字槽,k8 的 b1 槽闲置)----
    for i, r in enumerate(a):
        emit("st.shared.b32", f"[{D[3]}+{4 * i}]", r)
    for i, r in enumerate(b):
        emit("st.shared.b32", f"[{D[3]}+{16 + 4 * i}]", r)
    emit("bar.sync", "0")

    # ---- 读基址:A 行组 = 96g;B 列 2t = 192t;列 2t+1 = 192t+96 ----
    emit("mul.wide.u32", D[4], W[11], "96")
    emit("add.s64", D[5], D[0], D[4])             # A 行组基址
    emit("mul.wide.u32", D[6], W[12], "192")
    emit("add.s64", D[7], D[0], D[6])             # B 列 2t 基址
    emit("add.s64", D[8], D[7], "96")             # B 列 2t+1 基址

    # ---- 累加器 = C ----
    if atype == "f32":
        for i in range(4):
            emit("mov.f32", F[i], c[i])
    else:
        for w_i in range(2):
            emit("mov.b32", "{" + H[0] + ", " + H[1] + "}", c[w_i])
            emit("cvt.f32.f16", F[2 * w_i], H[0])
            emit("cvt.f32.f16", F[2 * w_i + 1], H[1])

    fi = 4                                        # f32 池游标(每轮复用)
    if not k8:
        # ============ k16 布局(lane 4g+t 存:A 行 g 的 (2t,2t+1) 与
        # (2t+8,2t+9) 列对、行 g+8 同列;B 列 g 的 (2t,2t+1)/(2t+8,2t+9))====
        for j in range(4):
            o = 24 * j
            for k, off in enumerate((0, 4, 8, 12)):
                emit("ld.shared.b32", W[16 + k], f"[{D[5]}+{o + off}]")
            emit("ld.shared.b32", W[20], f"[{D[7]}+{o + 16}]")
            emit("ld.shared.b32", W[21], f"[{D[7]}+{o + 20}]")
            emit("ld.shared.b32", W[22], f"[{D[8]}+{o + 16}]")
            emit("ld.shared.b32", W[23], f"[{D[8]}+{o + 20}]")

            unpack(W[16], F[fi], F[fi + 1])       # A 行 g,   k=2j,2j+1
            unpack(W[17], F[fi + 2], F[fi + 3])   # A 行 g+8, k=2j,2j+1
            unpack(W[18], F[fi + 4], F[fi + 5])   # A 行 g,   k=2j+8,2j+9
            unpack(W[19], F[fi + 6], F[fi + 7])   # A 行 g+8, k=2j+8,2j+9
            unpack(W[20], F[fi + 8], F[fi + 9])   # B 列 2t,   k=2j,2j+1
            unpack(W[21], F[fi + 10], F[fi + 11])  # B 列 2t,  k=2j+8,2j+9
            unpack(W[22], F[fi + 12], F[fi + 13])  # B 列 2t+1, k=2j,2j+1
            unpack(W[23], F[fi + 14], F[fi + 15])  # B 列 2t+1, k=2j+8,2j+9

            emit("fma.rn.f32", F[0], F[fi], F[fi + 8], F[0])
            emit("fma.rn.f32", F[0], F[fi + 1], F[fi + 9], F[0])
            emit("fma.rn.f32", F[0], F[fi + 4], F[fi + 10], F[0])
            emit("fma.rn.f32", F[0], F[fi + 5], F[fi + 11], F[0])
            emit("fma.rn.f32", F[1], F[fi], F[fi + 12], F[1])
            emit("fma.rn.f32", F[1], F[fi + 1], F[fi + 13], F[1])
            emit("fma.rn.f32", F[1], F[fi + 4], F[fi + 14], F[1])
            emit("fma.rn.f32", F[1], F[fi + 5], F[fi + 15], F[1])
            emit("fma.rn.f32", F[2], F[fi + 2], F[fi + 8], F[2])
            emit("fma.rn.f32", F[2], F[fi + 3], F[fi + 9], F[2])
            emit("fma.rn.f32", F[2], F[fi + 6], F[fi + 10], F[2])
            emit("fma.rn.f32", F[2], F[fi + 7], F[fi + 11], F[2])
            emit("fma.rn.f32", F[3], F[fi + 2], F[fi + 12], F[3])
            emit("fma.rn.f32", F[3], F[fi + 3], F[fi + 13], F[3])
            emit("fma.rn.f32", F[3], F[fi + 6], F[fi + 14], F[3])
            emit("fma.rn.f32", F[3], F[fi + 7], F[fi + 15], F[3])
    else:
        # ============ k8 布局(tools/sim_mma8.py 已对拍 numpy)============
        # lane 4g+t 存:a0 @0  = A 行 g   列 (2t,2t+1)
        #              a1 @4  = A 行 g+8 列 (2t,2t+1)
        #              b0 @16 = B 列 g   行 (2t,2t+1)
        # 读侧地址(相对 warp 槽基址 D0w,= 24*warp*32):
        #   A 行 g   k 对 (2j,2j+1):lane 4g+j   槽 → 96g + 24j + 0
        #   A 行 g+8 同列           → 同槽 @4            +4
        #   B 列 2t   k 对 (2j,2j+1):lane 8t+j   槽 → 192t + 24j + 16
        #   B 列 2t+1 同 k 对       → lane 8t+4+j 槽 → 192t + 96 + 24j + 16
        # D5 = 96g(A 基址)、D7 = 192t(B 列 2t)、D8 = 192t+96(列 2t+1)
        for j in range(4):
            o = 24 * j
            emit("ld.shared.b32", W[16], f"[{D[5]}+{o}+0]")    # A 行 g   k=2j,2j+1
            emit("ld.shared.b32", W[17], f"[{D[5]}+{o}+4]")    # A 行 g+8 同列
            emit("ld.shared.b32", W[20], f"[{D[7]}+{o}+16]")   # B 列 2t   k=2j,2j+1
            emit("ld.shared.b32", W[21], f"[{D[8]}+{o}+16]")   # B 列 2t+1 同 k 对

            unpack(W[16], F[fi], F[fi + 1])       # A 行 g   k=2j,2j+1
            unpack(W[17], F[fi + 2], F[fi + 3])   # A 行 g+8 同列
            unpack(W[20], F[fi + 8], F[fi + 9])   # B 列 2t
            unpack(W[21], F[fi + 10], F[fi + 11])  # B 列 2t+1

            # d0 = Σ A[g][k]·B[k][2t];d1 = …[2t+1];d2/d3 行 g+8
            emit("fma.rn.f32", F[0], F[fi], F[fi + 8], F[0])
            emit("fma.rn.f32", F[0], F[fi + 1], F[fi + 9], F[0])
            emit("fma.rn.f32", F[1], F[fi], F[fi + 10], F[1])
            emit("fma.rn.f32", F[1], F[fi + 1], F[fi + 11], F[1])
            emit("fma.rn.f32", F[2], F[fi + 2], F[fi + 8], F[2])
            emit("fma.rn.f32", F[2], F[fi + 3], F[fi + 9], F[2])
            emit("fma.rn.f32", F[3], F[fi + 2], F[fi + 10], F[3])
            emit("fma.rn.f32", F[3], F[fi + 3], F[fi + 11], F[3])
    # D 写回
    if atype == "f32":
        for i in range(4):
            emit("mov.f32", d[i], F[i])
    else:
        for w_i, (lo, hi) in enumerate(((0, 1), (2, 3))):
            emit("cvt.rn.f16.f32", B16[0], F[lo])
            emit("cvt.rn.f16.f32", B16[1], F[hi])
            emit("mov.b32", d[w_i], "{" + B16[0] + ", " + B16[1] + "}")
    return L


def pass_mma(entry, shared_used):
    """替换 entry 里的 mma 指令。shared_used: [bool] 透传是否注入了共享缓冲。"""
    body = []
    expanded = False
    for x in entry.body:
        if not isinstance(x, Inst) or not x.opcode.startswith("mma"):
            body.append(x)
            continue
        if x.opcode not in SHAPES:
            raise UnsupportedError(
                f"{entry.name} 行{x.line_no}: `{x.opcode}` — "
                f"该 mma 形状未实现(支持:{', '.join(sorted(SHAPES))})")
        if x.pred:
            raise UnsupportedError(
                f"{entry.name} 行{x.line_no}: 谓词化 mma 会导致 bar.sync "
                f"部分线程死锁,不支持")
        k_len, itype, atype = SHAPES[x.opcode]
        body.extend(_gen(x, k_len, itype, atype))
        expanded = True
    if expanded:
        head = list(DECLS)
        if not shared_used[0]:
            head.insert(0, f".shared .align 4 .b8 {SHARED_VAR}[{SHARED_TOTAL}];")
            shared_used[0] = True
        entry.body = head + body
    else:
        entry.body = body
