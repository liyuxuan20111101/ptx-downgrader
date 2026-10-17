# mma.m16n8k16 (f32.f16.f16.f32) layout tables + warp-expansion formula,
# verified against numpy in pure Python.
# Layout tables transcribed verbatim from the PTX ISA 9.0 document,
# section "Matrix Fragments for mma.m16n8k16".
# Logic: load A/B/C into 32 lanes' registers per the layout, run the
# "shared-memory exchange + FMA" expansion formula, write D back per the
# accumulator layout, and compare against numpy A@B+C. Any layout or
# formula error makes the reassembled D mismatch.
import numpy as np

M, N, K = 16, 8, 16


def fragments(lane):
    """(A halfword coords x8, B coords x4, C coords x4) for a lane."""
    g, t = lane >> 2, lane & 3
    # A: row = g (i<2 or 4<=i<6) else g+8; col = 2t+(i&1) (i<4) else +8
    A = []
    for i in range(8):
        row = g if (i < 2 or 4 <= i < 6) else g + 8
        col = 2 * t + (i & 1) + (8 if i >= 4 else 0)
        A.append((row, col))
    # B: row = 2t+(i&1) (i<2) else +8; col = g
    B = [(2 * t + (i & 1) + (8 if i >= 2 else 0), g) for i in range(4)]
    # C/D: row = g (i<2) else g+8; col = 2t+(i&1)
    C = [(g if i < 2 else g + 8, 2 * t + (i & 1)) for i in range(4)]
    return A, B, C


def simulate(A, Bm, C):
    """Simulate the whole warp per the PTX expansion. A:(16,16) f16,
    Bm:(16,8) f16, C:(16,8) f32."""
    # "shared memory": each lane stores 4 A words (2 halves each) + 2 B words
    smA = np.zeros((32, 4, 2), dtype=np.float32)  # [lane][word][lo/hi]
    smB = np.zeros((32, 2, 2), dtype=np.float32)
    Af = A.astype(np.float32)
    Bf = Bm.astype(np.float32)
    for lane in range(32):
        fa, fb, _ = fragments(lane)
        for w in range(4):
            smA[lane][w] = [Af[fa[2 * w]], Af[fa[2 * w + 1]]]
        for w in range(2):
            smB[lane][w] = [Bf[fb[2 * w]], Bf[fb[2 * w + 1]]]

    D = np.zeros((M, N), dtype=np.float32)
    for lane in range(32):
        fa, fb, fc = fragments(lane)
        g, t = lane >> 2, lane & 3
        acc = [np.float32(C[fc[i]]) for i in range(4)]
        # Mirrors the emitted PTX: lane group 4g+j word0/2 = row g,
        # k=2j,2j+1 and k=2j+8,2j+9; word1/3 = row g+8 same columns;
        # B group 8t+j word0/1 = column 2t, same k pairs; column 2t+1
        # uses lanes 8t+4+j.
        for j in range(4):
            Aw = [[smA[4 * g + j][0][0], smA[4 * g + j][0][1]],   # row g   k=2j,2j+1
                  [smA[4 * g + j][2][0], smA[4 * g + j][2][1]],   # row g   k=2j+8,2j+9
                  [smA[4 * g + j][1][0], smA[4 * g + j][1][1]],   # row g+8 k=2j,2j+1
                  [smA[4 * g + j][3][0], smA[4 * g + j][3][1]]]   # row g+8 k=2j+8,2j+9
            Bc0 = [[smB[8 * t + j][0][0], smB[8 * t + j][0][1]],  # col 2t   k=2j,2j+1
                   [smB[8 * t + j][1][0], smB[8 * t + j][1][1]]]  # col 2t   k=2j+8,2j+9
            Bc1 = [[smB[8 * t + 4 + j][0][0], smB[8 * t + 4 + j][0][1]],
                   [smB[8 * t + 4 + j][1][0], smB[8 * t + 4 + j][1][1]]]
            # d0 = sum_k A[g][k]*B[k][2t]; d1 uses column 2t+1; d2/d3 row g+8
            for (a, b) in zip(Aw[0], Bc0[0]):
                acc[0] = np.float32(acc[0] + np.float32(a * b))
            for (a, b) in zip(Aw[1], Bc0[1]):
                acc[0] = np.float32(acc[0] + np.float32(a * b))
            for (a, b) in zip(Aw[0], Bc1[0]):
                acc[1] = np.float32(acc[1] + np.float32(a * b))
            for (a, b) in zip(Aw[1], Bc1[1]):
                acc[1] = np.float32(acc[1] + np.float32(a * b))
            for (a, b) in zip(Aw[2], Bc0[0]):
                acc[2] = np.float32(acc[2] + np.float32(a * b))
            for (a, b) in zip(Aw[3], Bc0[1]):
                acc[2] = np.float32(acc[2] + np.float32(a * b))
            for (a, b) in zip(Aw[2], Bc1[0]):
                acc[3] = np.float32(acc[3] + np.float32(a * b))
            for (a, b) in zip(Aw[3], Bc1[1]):
                acc[3] = np.float32(acc[3] + np.float32(a * b))
        for i, (row, col) in enumerate(fc):
            D[row][col] = acc[i]
    return D


def main():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((M, K)).astype(np.float16)
    Bm = rng.standard_normal((K, N)).astype(np.float16)
    C = rng.standard_normal((M, N)).astype(np.float32)

    D = simulate(A, Bm, C)
    want = A.astype(np.float32) @ Bm.astype(np.float32) + C
    err = np.abs(D - want).max()
    rel = err / np.abs(want).max()
    print(f"max abs err = {err:.2e}, rel = {rel:.2e}")
    ok = rel < 1e-5
    print("layout + expansion formula:", "VERIFIED" if ok else "FAILED!")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
