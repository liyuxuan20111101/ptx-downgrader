# mma.m16n8k8 (f32.f16.f16.f32) layout + warp-expansion formula, verified
# against numpy in pure Python.
# Layout per PTX ISA 9.0, section "Matrix Fragments for mma.m16n8k8".
# IMPORTANT: in that section's table, a0..a3 / b0..b1 are HALFWORD tags,
# not registers. The real register footprint is:
#   A: two .f16x2 registers —
#        a0 = row g,   cols (2t, 2t+1);  a1 = row g+8, cols (2t, 2t+1)
#   B: one .f16x2 register —
#        b0 = (k=2t, n=g), (k=2t+1, n=g)
#   C/D: same as k16 (row = g / g+8, col = 2t + (i & 1))
# Tiling completeness (every matrix cell covered exactly once by the 32
# lanes) was verified separately and pins this reading down.
# Shared-memory plan: 24-byte lane slot = a0 @0, a1 @4, b0 @16.
# Read side (per lane with its own g, t):
#   A row g    k-pair (2j,2j+1) -> lane 4g+j   slot @0
#   A row g+8  same columns    -> lane 4g+j   slot @4
#   B col 2t   k-pair (2j,2j+1) -> lane 8t+j   slot @16
#   B col 2t+1 same k-pair     -> lane 8t+4+j slot @16
import numpy as np

M, N, K = 16, 8, 8


def fragments(lane):
    g, t = lane >> 2, lane & 3
    A = [(g, 2 * t), (g, 2 * t + 1), (g + 8, 2 * t), (g + 8, 2 * t + 1)]
    B = [(2 * t, g), (2 * t + 1, g)]
    C = [(g if i < 2 else g + 8, 2 * t + (i & 1)) for i in range(4)]
    return A, B, C


def simulate(A, Bm, C):
    smA = np.zeros((32, 2, 2), dtype=np.float32)   # [lane][reg][lo/hi]
    smB = np.zeros((32, 2), dtype=np.float32)      # [lane][lo/hi]
    Af = A.astype(np.float32)
    Bf = Bm.astype(np.float32)
    for lane in range(32):
        fa, fb, _ = fragments(lane)
        for w in range(2):
            smA[lane][w] = [Af[fa[2 * w]], Af[fa[2 * w + 1]]]
        smB[lane] = [Bf[fb[0]], Bf[fb[1]]]

    D = np.zeros((M, N), dtype=np.float32)
    for lane in range(32):
        fa, fb, fc = fragments(lane)
        g, t = lane >> 2, lane & 3
        acc = [np.float32(C[fc[i]]) for i in range(4)]
        for j in range(4):                    # k pair (2j, 2j+1)
            a0 = smA[4 * g + j][0]            # A row g    k=2j,2j+1
            a1 = smA[4 * g + j][1]            # A row g+8  same columns
            b0 = smB[8 * t + j]               # B col 2t   k=2j,2j+1
            b1 = smB[8 * t + 4 + j]           # B col 2t+1 same k pair
            acc[0] = np.float32(acc[0] + a0[0] * b0[0] + a0[1] * b0[1])
            acc[1] = np.float32(acc[1] + a0[0] * b1[0] + a0[1] * b1[1])
            acc[2] = np.float32(acc[2] + a1[0] * b0[0] + a1[1] * b0[1])
            acc[3] = np.float32(acc[3] + a1[0] * b1[0] + a1[1] * b1[1])
        for i, (row, col) in enumerate(fc):
            D[row][col] = acc[i]
    return D


def main():
    rng = np.random.default_rng(11)
    A = rng.standard_normal((M, K)).astype(np.float16)
    Bm = rng.standard_normal((K, N)).astype(np.float16)
    C = rng.standard_normal((M, N)).astype(np.float32)
    D = simulate(A, Bm, C)
    want = A.astype(np.float32) @ Bm.astype(np.float32) + C
    err = np.abs(D - want).max()
    print(f"max abs err = {err:.2e}")
    ok = err / np.abs(want).max() < 1e-5
    print("k8 layout + expansion formula:", "VERIFIED" if ok else "FAILED!")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
