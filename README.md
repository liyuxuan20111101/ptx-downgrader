# ptx-downgrader

[![CI](https://github.com/liyuxuan20111101/ptx-downgrader/actions/workflows/ci.yml/badge.svg)](https://github.com/liyuxuan20111101/ptx-downgrader/actions/workflows/ci.yml)

**Run CUDA 13-era PTX on pre-Turing GPUs — or any GPU whose driver stopped receiving native PTX — by degrading modern PTX to the target's ISA. Targets: sm_61 (Pascal/GTX 10-series), sm_70, sm_75, sm_80, sm_86 (Ampere/RTX 30-series), sm_89 (Ada). Pure software, no driver patching, no signature hacks.**

When NVIDIA ended Pascal support (driver branch R580 was the last; CUDA 13 removed `sm_61` compilation targets), applications shipping only modern PTX or SASS stopped working on GTX 10-series GPUs. This project implements the missing link:

```
CUDA 13 fatbin (sm_75/80/90 PTX sections)
        │  cuobjdump extraction
        ▼
  modern PTX (ISA 8.x, sm_90)
        │  ptx-downgrader  ◄─── this project
        ▼
  PTX ISA 7.0, .target sm_61
        │  ptxas / driver JIT
        ▼
  runs correctly on your GTX 1070
```

**Verified end to end on real CUDA 13.4 binaries** (llama.cpp `ggml-cuda.dll`): a real kernel extracted from the fatbin, downgraded, and executed on a GTX 1070 under driver 472.12 matched the numpy reference implementation bit-for-bit (error 0.00e+00), and matched the non-degraded reference build exactly.

## What it does

The `downgrader` package rewrites PTX at the text/IR level. Every transform is semantics-preserving:

| Transform | Why it's needed |
|---|---|
| `.version 8.x → 7.0`, `.target sm_XX → sm_61` | Old drivers' JIT only accepts local-arch PTX |
| `mma.sync.m16n8k16/m16n8k8` (6 dtype shapes) → shared-memory exchange + FMA sequences | Tensor-core instructions have no sm_61 hardware; each mma becomes a block-wide data exchange through shared memory plus 32–64 scalar FMAs |
| `cp.async` → `ld.global` + `st.shared` (+ zero-fill for `src-size`, `bar.sync` for waits) | Asynchronous copies need sm_80 hardware |
| `mbarrier` init/arrive/wait → conservative `bar.sync` or removal | Same |
| `tanh.approx.f32/f16` → 5-instruction software sequence (`ex2`/`rcp`/`fma`) | sm_75 instruction, 3271 occurrences in llama.cpp; GPU-verified max error 5.96e-08 |
| `bmsk.clamp.b32` → `shr` + constant-mask `and` | sm_100 bitfield mask (1694 occurrences in llama.cpp) |
| `griddepcontrol` removal | Programmatic dependent launch is a scheduling hint; removal is safe |
| Mixed-width addresses `[base+reg32]` → materialize + `cvt.u64.u32` + `add.s64` | PTX 7.0 rejects 64-bit base + 32-bit register offset forms |
| Special registers in ALU/addresses (`%tid.x` as operand) → materialize via `mov` | PTX 7.0 restriction |
| Opcode fixes: `dp4a.lo` qualifier, `mov.f16` immediates, cache-hint stripping (`L1::evict_*`) | ISA 7.0 syntax differences |
| int8 `mma.sync` (m16n8k32/m16n8k16/m8n8k16, s32 acc) → `dp4a` sequences | Quantized-inference workhorse; verified 128/128 exact on GTX 1070 |
| Rejection with actionable errors: `wgmma`, `tcgen05`, TMA family, `ldmatrix/stmatrix`, `wmma`, clusters | No software equivalent at acceptable cost (for now) |

Everything runs **entirely in user space**: no driver modification, no kernel-mode code, no signature patching. The output is plain PTX that ptxas compiles and the stock driver loads.

## Requirements

- Python 3.10+ with `numpy` (`pip install -r requirements.txt`)
- A Pascal GPU (GTX 10-series, sm_61) for on-GPU verification
- Any driver that supports the GPU — the project was developed entirely on driver 472.12 (2021) without touching it

## Setup

```bash
git clone <this repo>
cd ptx-downgrader
pip install -r requirements.txt
python setup_tools.py        # downloads cuobjdump + ptxas (~30 MB) from NVIDIA's official redist server
```

`setup_tools.py` fetches NVIDIA's freely redistributable tool components (the same way the CUDA installer does), validates every URL against a single-host allow-list, blocks redirects and private-address DNS resolution, and guards zip extraction against path traversal.

## Usage

### CLI: downgrade a PTX file

```bash
python -m downgrader input.ptx -o output.sm61.ptx               # Pascal (default)
python -m downgrader input.ptx --target sm_86                   # Ampere / RTX 30-series
# validates output with ptxas automatically (when ptxas is present)
```

### Compatibility survey of a real fatbin

```bash
# 1. extract PTX sections from a CUDA 13 binary
tools/cuda_cuobjdump-windows-x86_64-13.0.85-archive/bin/cuobjdump.exe -ptx app.dll > full_ptx.txt

# 2. survey every kernel: downgrade, classify pass/reject, histogram rejection reasons
python tools/survey.py full_ptx.txt
```

Measured on llama.cpp b11574 (CUDA 13.4, `ggml-cuda.dll`, 105 unique kernels):

```
sm_75:  203/222 kernels downgradable (91.4%)
sm_80:  203/221 (91.9%)
sm_90:  203/221 (91.9%)
```

Remaining rejects are the int8 `mma` family (`s32.s8.s8.s32` shapes — layout work pending) and `ldmatrix/stmatrix` (fragment-layout rearrangement, pending).

### GPU verification suites

```bash
python m1/test_e2e.py      # 5 kernels covering every transform, numpy cross-checked
python m2/verify_fwht.py   # real llama.cpp kernel, triple verification (needs tools/full_ptx.txt)
python tools/sim_mma.py    # fragment-layout formulas vs numpy (pure CPU)
python tools/sim_mma8.py
```

## Repository layout

```
downgrader/     the library + CLI (parser, rule engine, mma expander, ptxas validator)
m0/             feasibility spike: driver-JIT pipeline proof (3 kernels, zero toolchain)
m1/             full e2e test suite: mixed kernels, mma k16/k8, cp.async, GPU golden tests
m2/             real-world verification: llama.cpp fwht kernel, triple-checked on GPU
tools/          survey/audit scripts, mma simulators (binaries excluded — run setup_tools.py)
DEVLOG.md       full development log with every bug found and mechanism discovered
```

## Key findings from development (details in DEVLOG.md)

- Old-driver JIT (472.x) **only loads `.target sm_61` PTX** — target rewriting isn't optional, it's the enabling step.
- PTX 7.0 rejects `[64-bit base + 32-bit register]` addresses; CUDA 13 compilers emit this form constantly.
- Predicate-guarded `wait` instructions must lose their predicate when downgraded to `bar.sync`, or the block deadlocks.
- The `mma.m16n8k8` fragment documentation's `a0..a3` labels are **halfword tags, not registers** — the actual footprint is 2 registers for A, 1 for B.

## Known limits

- cuBLAS/cuDNN ship SASS-only (no PTX): engines depending on them can't be downgraded. Self-contained kernel engines (llama.cpp with `-DGGML_CUDA_NO_CUBLAS`, or builds using `-arch=compute_61`) are the target use case.
- Downgraded tensor-core code runs at scalar-FMA speed; expect a fraction of native mma throughput.
- FMA accumulation order differs from hardware mma — results match within float tolerance, not bit-exact (except where the reference is also order-exact).
- `ldmatrix`/`redux.sync`/`elect.sync` expansion passes are tracked for v0.4.

## License

MIT for everything in this repository. NVIDIA cuobjdump/ptxas binaries are downloaded separately by `setup_tools.py` and remain under the NVIDIA CUDA Toolkit EULA.
