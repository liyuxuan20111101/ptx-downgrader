# mma.m16n8k32 (s32.s8.s8.s32) 布局表 + 展开公式模拟验证。
# 布局逐字来自 PTX ISA 9.0 "Matrix Fragments for mma.m16n8k32" (.s8/.u8):
#   A: 4 个 .b32 寄存器(每寄存器 4 个 s8,a0..a15 为半字级标签):
#      row = g        for i where 0 <= i < 4 || 8 <= i < 12
#      row = g + 8    otherwise
#      col = 4t + (i & 3)      for i < 8
#      col = 4t + (i & 3) + 16 for i >= 8
#      → 每寄存器(4 个连续半字标签):
#        w0 = (g,     4t..4t+3)   w1 = (g,     4t+16..4t+19)
#        w2 = (g+8,   4t..4t+3)   w3 = (g+8,   4t+16..4t+19)
#   B: 2 个 .b32 寄存器(b0..b7 半字级):
#      row = 4t + (i & 3)       for i < 4
#      row = 4t + (i & 3) + 16  for i >= 4
#      col = g
#      → 寄存器 b0 = (4t..4t+3, g),b1 = (4t+16..4t+19, g)
#   C/D: 4 x .s32,row = g/g+8,col = 2t+(i&1) —— 与 f16 系相同。
#
# 降级公式:lane 组 4g+j 的 w0 覆盖行 g 的 k=4t'..:对目标 lane(g,t),
# 它需要行 g 全部 32 个 k。行 g 的 k=4j..4j+3 在 lane 4g+j 的 w0;
# k=4j+16..4j+19 在 lane 4g+j 的 w1。B 列 2t 的 k=4j..4j+3 在
# lane 8(2t)+j?—— B 坐标:lane l 的 b0 = (k=4(l&3)..+3, n=l>>2)。
# 列 n=2t 的 k=4j 组在 lane 8(2t)+j;列 2t+1 在 lane 8(2t+1)+j = 8t+8+j。
# 每 lane 组 4 个 k → 32 个 k 需 8 组?k32 = 32 个 k,每组 4 → 8 组,
# 但 lane 4g+j 只有 j∈0..3 四个 j → 覆盖 k=0..15(行 g 前半)?
# 不:lane 4g+j 的 w0 = 行 g 的列 (4j..4j+3) 其中 j = l&3 = t。
# 对固定 lane(g,t):它持有行 g 的列 (4t..4t+3) 和 (4t+16..4t+19)。
# 行 g 的完整 32 个 k 分散在 lane 4g+0..3(t=0..3 各 4+4 个,即 16+16)
# → t=0..3 覆盖列 0..15(t*4..t*4+3)与 16..31(+16)✓ 全覆盖!
# 读侧(目标 lane 的 g,t):行 g 的 k=4j+r(j∈0..3, r∈0..3)在
# lane 4g+j 的 w0 的第 r 字节;行 g 的 k=16+4j+r 在 lane 4g+j 的 w1;
# 行 g+8 同列在 lane 4g+j 的 w2/w3。
import numpy as np

M, N, K = 16, 8, 32


def fragments(lane):
    """lane 的 (A 字节坐标 x16, B 字节坐标 x8, C 坐标 x4)。"""
    g, t = lane >> 2, lane & 3
    A = []
    for w in range(4):                    # 4 个 b32 寄存器
        row = g if (w < 2) else g + 8
        base = 4 * t + 16 * (w & 1)
        for byte in range(4):
            A.append((row, base + byte))
    B = []
    for w in range(2):                    # 2 个 b32 寄存器
        base = 4 * t + 16 * w
        for byte in range(4):
            B.append((base + byte, g))
    C = [(g if i < 2 else g + 8, 2 * t + (i & 1)) for i in range(4)]
    return A, B, C


def simulate(A, Bm, C):
    """按共享内存交换 + dp4a 等价(纯 python 用字节乘加)模拟。
    A:(16,32) int8, Bm:(32,8) int8, C:(16,8) s32。"""
    smA = np.zeros((32, 4, 4), dtype=np.int32)   # [lane][word][byte]
    smB = np.zeros((32, 2, 4), dtype=np.int32)
    for lane in range(32):
        fa, fb, _ = fragments(lane)
        for w in range(4):
            for by in range(4):
                smA[lane][w][by] = A[fa[4 * w + by]]
        for w in range(2):
            for by in range(4):
                smB[lane][w][by] = Bm[fb[4 * w + by]]

    D = np.zeros((M, N), dtype=np.int32)
    for lane in range(32):
        fa, fb, fc = fragments(lane)
        g, t = lane >> 2, lane & 3
        acc = [int(C[fc[i]]) for i in range(4)]
        # dp4a 对:每 lane 组 4g+j 的 w0(行 g,k=4j..4j+3)配
        # B 列 2t 的 k=4j..4j+3(lane 8t+j 的 b0);
        # w1(行 g,k=16+4j..)配 lane 8t+j 的 b1(k=16+4j..)。
        for j in range(4):
            a0 = smA[4 * g + j][0]        # 行 g   k=4j..4j+3
            a1 = smA[4 * g + j][1]        # 行 g   k=16+4j..+3
            a2 = smA[4 * g + j][2]        # 行 g+8 k=4j..4j+3
            a3 = smA[4 * g + j][3]        # 行 g+8 k=16+4j..+3
            b0 = smB[8 * t + j][0]        # 列 2t   k=4j..4j+3
            b1 = smB[8 * t + j][1]        # 列 2t   k=16+4j..+3
            b0c = smB[8 * t + 4 + j][0]   # 列 2t+1 k=4j..4j+3 (lane 8t+4+j)
            b1c = smB[8 * t + 4 + j][1]   # 列 2t+1 k=16+4j..+3
            dp = lambda x, y: int(sum(int(px) * int(py) for px, py in zip(x, y)))
            acc[0] += dp(a0, b0) + dp(a1, b1)
            acc[1] += dp(a0, b0c) + dp(a1, b1c)
            acc[2] += dp(a2, b0) + dp(a3, b1)
            acc[3] += dp(a2, b0c) + dp(a3, b1c)
        for i, (row, col) in enumerate(fc):
            D[row][col] = acc[i]
    return D


def main():
    rng = np.random.default_rng(13)
    A = rng.integers(-128, 128, (M, K)).astype(np.int8)
    Bm = rng.integers(-128, 128, (K, N)).astype(np.int8)
    C = rng.integers(-1000, 1000, (M, N)).astype(np.int32)

    D = simulate(A, Bm, C)
    A32 = A.astype(np.int32)
    B32 = Bm.astype(np.int32)
    want = np.zeros((M, N), dtype=np.int64)
    for i in range(M):
        for nn in range(N):
            want[i, nn] = C[i, nn] + int(np.dot(A32[i], B32[:, nn]))
    err = np.abs(D - want).max()
    print(f"max abs err = {err}")
    ok = err == 0
    print("k32 s8 布局 + 展开公式:", "VERIFIED" if ok else "FAILED!")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
