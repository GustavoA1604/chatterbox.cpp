# ggml patches for Chatterbox

`ggml` is vendored as a pristine upstream clone (see the top-level
[`README.md`](../README.md)), so any fixes we need in it live here as
standalone patches and are applied after the clone.

Two patches ship today:

1. [`ggml-metal-chatterbox-ops.patch`](#ggml-metal-chatterbox-opspatch) —
   fills gaps in the Metal backend (missing `diag_mask_inf`, front-pad
   `PAD`, scalar `conv_transpose_1d`, `MUL_MAT + ADD(+ADD)` fusion).
   Apple-only; harmless no-op on CPU / CUDA / Vulkan builds.
2. [`ggml-cuda-chatterbox-ops.patch`](#ggml-cuda-chatterbox-opspatch) —
   two related ggml-cuda fixes for chatterbox-style workloads:
   (a) replaces the scalar `conv_transpose_1d` CUDA kernel
   (one thread per output pixel scanning all `IC * IL` inputs with a
   skip conditional) with a warp-cooperative version that narrows the
   input range and reduces across `IC` via `__shfl_xor_sync` —
   ~40× faster on HiFT-shaped inputs;
   (b) adds an opt-in `GGML_CUDA_FORCE_GRAPHS=1` env var that bypasses
   the strict warmup check in `ggml_backend_cuda_graph_compute` so
   autoregressive workloads with growing-KV cache (chatterbox T3,
   GPT-style step-decode) actually hit the graph cache —
   ~7-14 % faster T3 step depending on prompt length.
   Both active whenever `-DGGML_CUDA=ON` is in the build; force-graphs
   is opt-in via env var, conv_transpose patch is unconditional.

`scripts/setup-ggml.sh` applies both in order; the patches stack
cleanly on the same pinned upstream commit.

## Apply

The top-level [`scripts/setup-ggml.sh`](../scripts/setup-ggml.sh) does
everything for you:

```bash
# From the repo root.  Clones ggml if needed, checks out the pinned
# commit, and applies every patch under patches/.  Idempotent —
# re-running is a no-op.
./scripts/setup-ggml.sh
```

Then configure + build as usual.  Pick the backend flags for your
platform:

```bash
# Apple Silicon — picks up the Metal patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON

# Linux / Windows + NVIDIA — picks up the CUDA conv_transpose_1d patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON

cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
```

If you'd rather run the steps by hand (e.g. to pin a different
upstream commit), the script is effectively:

```bash
git clone https://github.com/ggml-org/ggml.git ggml
cd ggml && git checkout $GGML_COMMIT
git apply ../patches/ggml-metal-chatterbox-ops.patch
git apply ../patches/ggml-cuda-chatterbox-ops.patch
```

`GGML_COMMIT` lives at the top of `scripts/setup-ggml.sh` as the
single source of truth — bump it when re-generating the patches
against a newer upstream ggml.  To confirm everything applied
cleanly:

```bash
(cd ggml && git status --short)
# Expected: 7 modified files under ggml/src/ggml-metal/
#           3 modified files under ggml/src/ggml-cuda/
```

CPU-only or Vulkan builds get the pinned commit but no useful patch
work: both targeted backends (Metal, CUDA) are additive and harmless
when their compile flag is off.

## `ggml-metal-chatterbox-ops.patch`

Base commit: `58c3805` (`sync : llama.cpp`, 2026-04-09).

Fixes three gaps in ggml-metal that make Chatterbox unusable or very slow
on Metal:

| Symptom                                       | Root cause in ggml-metal                          | What this patch does                                           |
|-----------------------------------------------|---------------------------------------------------|----------------------------------------------------------------|
| T3 crashes: `unsupported op 'DIAG_MASK_INF'`  | No op entry / no kernel                            | Adds `kernel_diag_mask_inf_f32`, dispatcher, `supports_op` case|
| S3Gen crashes: `unsupported op 'PAD'` when any front-pad (`lp0..lp3`) is non-zero | Kernel only supports tail padding; `supports_op` rejects non-zero front pads | Extends `kernel_pad_f32` + `ggml_metal_kargs_pad` to honour `lp0..lp3` and drops the rejection  |
| HiFT decode is ~100× slower than CPU          | `kernel_conv_transpose_1d` is scalar: 1 thread per output pixel iterating over *all* `IC * IL` inputs, with most of the work inside a conditional | Tighten the input-position range to the few that contribute (`i_min..i_max`) and parallelise `IC` across a 32-thread simdgroup with `simd_sum` reduction |
| T3 step does `mul_mv + bin_fuse(add)` / `mul_mv + bin_fuse(add+add)` per linear layer | `mul_mv` and the following bias / bias+residual adds are separate Metal kernels even though Vulkan fuses the same patterns (`ggml_vk_can_fuse` + `Fuse0` / `Fuse1` shader bindings) | Fuse `mul_mat + add(bias)` and `mul_mat + add(bias) + add(residual)` for the Q-variant mat-vec kernels (Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0) via two function constants (`FC_mul_mv_has_bias`, `FC_mul_mv_has_residual`) and a `helper_mv_add_bias<NR0>` post-pass.  The op encoder tries `{MUL_MAT, ADD, ADD}` first and falls back to `{MUL_MAT, ADD}`; `n_fuse` tells the dispatcher how many nodes to consume |

Measured on M3 Ultra, `hift_decode` at HiFT-realistic shapes:
- Before: ~15 000 ms
- After:    ~350 ms (≈ 40× speedup; end-to-end `gen_RTF` goes from
  unusable → 0.19 on F16)

Correctness is validated against the ggml CPU backend by the
`test-metal-ops` binary built in the parent repo (Metal builds only).
Run it after rebuilding:

```bash
./build/test-metal-ops
# Expected: "diag_mask_inf / pad_ext / conv_transpose_1d: PASS"
```

## Dropping the patch

If upstream ggml merges equivalent fixes, delete the patch file and
remove the corresponding entry from the `PATCHES=(…)` array in
`scripts/setup-ggml.sh`.  The C++ side of Chatterbox uses only ops
supported by every backend, so nothing else needs to change.

No patch is needed for CPU / Vulkan — those backends already handle
every op Chatterbox emits at production speed.

## `ggml-cuda-chatterbox-ops.patch`

Base commit: same as above (`58c3805`).

Two ggml-cuda changes targeted at chatterbox's pipeline shapes:

1. **`conv_transpose_1d_kernel` rewrite** — ~40× faster on the HiFT
   vocoder graphs Chatterbox emits.  Always on whenever `GGML_CUDA=ON`.
2. **`GGML_CUDA_FORCE_GRAPHS=1` env var** — bypass the strict warmup
   check so autoregressive (T3-step / growing-KV) workloads can use
   CUDA Graphs.  Opt-in.

Both modify only `ggml/src/ggml-cuda/`; no other backend is touched
and the changes are no-ops when CUDA isn't built.  Correctness
preserved end-to-end (audio output identical at -57 dBFS / SNR 58 dB
vs the stock kernel for the conv_transpose change; bit-identical for
the force-graphs change since it only changes scheduling not math).

### Part 1: warp-cooperative `conv_transpose_1d_kernel`

The stock kernel is the textbook "1 thread per output pixel"
implementation:

```cuda
int idx = global_index % dst_ne0;          // ol
int oc  = global_index / dst_ne0;          // out channel
for (int ic = 0; ic < IC; ic++) {
    for (int i = 0; i < IL; i++) {         // <-- hot inner loop
        if (!(idx >= i*s0 && idx < i*s0 + K)) continue;   // most iters skip
        v += kernel[ki, oc, ic] * input[i, ic];
    }
}
```

Two perf killers:

1. **The `i` loop scans all `IL` positions** even though only
   `K/s0 + 1` of them contribute (KS=16, s0=8 ⇒ ≤ 2 of 100+).
2. **No parallelism across `IC`** — the IC reduction is fully
   serial within a single thread.

For HiFT the typical first upsample layer has `IC=512, IL=104, KS=16,
s0=8, OL=840, OC=256`. Each thread does `512 * 104 = 53 248` inner
iterations, of which only ~2 do real work. With `OL*OC = 215 040`
output pixels that's ~11.4 billion no-op iterations per call — so
of the 4 conv_transpose_1d calls in the graph, the largest takes
~100 ms on RTX 5090 in the stock kernel.

The patched kernel:

1. Switches to **one CUDA warp per output pixel**: grid
   `(OL, OC, 1)`, block `(32, 1, 1)`. Same as the Metal-patch
   kernel design (one threadgroup per output, 32-wide simdgroup
   reduction).
2. Computes `i_start, i_end` analytically from
   `i*s0 + ki = ol, 0 ≤ ki < K, 0 ≤ i < IL`, so the `i` loop
   only iterates over the contributing range. No more skip
   conditional in the hot loop.
3. **Parallelises the `IC` reduction across the warp** — each
   thread handles a strided slice `ic = tid, tid+32, …` and
   accumulates a partial sum.
4. Reduces across the 32-lane warp with `__shfl_xor_sync` and
   thread 0 writes the final sum.

The host-side wrapper signature, the `ggml_cuda_op_conv_transpose_1d`
entry point, and the supported shape constraints (contiguous src0/src1,
`ne1 == ne3 == 1`) are unchanged. `CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE`
in `conv-transpose-1d.cuh` drops from 256 to 32 (one warp).

### Measured impact (RTX 5090 + CUDA 12.0 + driver 590.48, Turbo Q4_0)

`build/chatterbox` on the same prompt + seed across 5 runs (median of
runs 2–5, NVIDIA driver cache warm):

| Stage         | Stock kernel | Patched | Speedup |
|---------------|-------------:|--------:|--------:|
| `[hift_decode]`     | 149.5 ms     | 22 ms   | **6.8×** |
| `[hift_total]`      | 144.7 ms     | 30 ms   | **4.7×** |
| `S3GEN_INFER_MS`    | 280 ms       | 173 ms  | **1.6×** |
| Wall (T3 + S3Gen)   | 442 ms       | 337 ms  | **1.3×** |
| RTF (S3Gen / audio) | 0.15         | 0.09    |   —      |

Per-op profile from `nsys profile --trace=cuda` (4 calls of
`conv_transpose_1d_kernel` in a single warm S3Gen run):

| Field           | Stock    | Patched  |
|-----------------|---------:|---------:|
| Total GPU time  | 135.98 ms |  3.21 ms |
| Avg per call    |  33.99 ms |  802 µs  |
| Max single call | 101.34 ms |  1.41 ms |
| **Speedup**     |     —     | **42×**  |

For comparison, ggml-vulkan's stock `conv_transpose_1d.comp` runs
the same shapes in ~3.4 ms on the same hardware (FINDINGS.md §3.2);
the patched CUDA kernel is now within noise of that.

T3 (autoregressive decoder, no `conv_transpose_1d`) is unchanged at
~163 ms / 43 tokens — confirming the saving is localised to HiFT
where it should be.

### Correctness

The patch changes the **floating-point reduction order** (warp-strided
IC accumulation + `__shfl_xor_sync` cross-lane sum vs the stock
kernel's serial accumulation), so output is not bit-identical. End-to-
end audio comparison on the same seed and prompt:

| Metric                           | Value           |
|----------------------------------|-----------------|
| Sample count                     | 44 160 (same)   |
| Max abs sample diff              | 0.001221 (-58 dBFS) |
| Mean abs sample diff             | 0.000016        |
| RMS error                        | 0.000043        |
| SNR (signal vs diff)             | **58.5 dB**     |

i.e. inaudible.  This is the same kind of FP-order variance that the
Metal patch's simdgroup-sum reduction introduces.

### Part 2: `GGML_CUDA_FORCE_GRAPHS` opt-in

#### Background

Default ggml-cuda graph caching is keyed on the first node of the
incoming `ggml_cgraph_t` and gated by a 2-call warmup check
(`ggml_cuda_graph_update_required`) that requires every property of
every node to be byte-identical between calls.  That's the right
default for llama.cpp-style workloads where each call sends the same
graph object with stable data pointers.

Chatterbox's T3 step builds a fresh-but-identical-topology cgraph
per token *with growing K/V views* (`L = n_past + 1`, see
`build_step_graph` in `src/main.cpp`).  Each call therefore looks
"different" to the strict warmup check — `K`'s `ne[1]` grows by 1,
view offsets shift, etc. — so `warmup_complete` keeps resetting and
the captured cudaGraph is never actually used.  The cost manifests
as a multi-ms-per-token gap to e.g. ggml-vulkan, whose dispatcher
has lower per-launch overhead.

#### What the env var does

When `GGML_CUDA_FORCE_GRAPHS=1` is set, the early-exit branch in
`ggml_backend_cuda_graph_compute` is replaced with:

```cpp
use_cuda_graph = true;
cuda_graph_update_required = properties_changed || graph->instance == nullptr;
```

i.e. *always* try the captured-graph path; rely on the existing
`cudaGraphExecUpdate` (and re-instantiate-on-failure) wiring in
`ggml_cuda_graph_update_executable` to absorb per-call differences.
Default behaviour is unchanged when the env var is unset.

#### Measured impact (RTX 5090 + CUDA 12.0, Turbo Q4_0, fresh process median)

| Prompt length (tokens) | Default (no graphs) | `FORCE_GRAPHS=1` | Δ T3   | Δ %   |
|------------------------:|--------------------:|-----------------:|-------:|------:|
|                  19    |        120 ms       |       113 ms     | -7 ms  | -6 %  |
|                  43    |        163 ms       |       150 ms     | -13 ms | -8 %  |
|                 157    |        384 ms       |       332 ms     | -52 ms | -14 % |
|                 231    |        523 ms       |       453 ms     | -70 ms | -13 % |

Per-token saving is constant at ~0.3 ms/token across prompt lengths
— consistent with replacing `~30 cudaLaunchKernel` calls per token
(at ~4 µs each) with a single `cudaGraphLaunch` + `cudaGraphExecUpdate`
(at ~50 µs total).  Modest but free, and S3Gen / CFM are unchanged.

#### Correctness

Audio output is **bit-identical** with vs without the env var
(`md5sum` matches across `tokens=43` and `tokens=231` runs).  The
patch only changes graph orchestration, not kernel math; the inner
kernels still run the same arguments via `cudaGraphExecUpdate`.

#### When NOT to set it

- Workloads that genuinely change topology between calls (e.g. a
  model that loads variable-length subgraphs per request).  In that
  case `cudaGraphExecUpdate` would fail and trigger re-instantiation
  every call, which is *slower* than the no-graphs fallback. Stick
  with the default.
- Models where each call's KV-cache properties are stable
  (llama.cpp's default decode loop): the regular warmup machinery
  already kicks in and `FORCE_GRAPHS` is at most a no-op.

### Caveats / follow-ups

* **Standalone CUDA Graphs (without FORCE_GRAPHS) are net-negative
  for this pipeline.** Independent finding from the same
  investigation: building with `-DGGML_CUDA_GRAPHS=ON` and running
  without `FORCE_GRAPHS=1` adds ~7 ms to `[cfm_total]` (capture
  overhead vs launch saving) and is identical on T3.  Either build
  with graphs OFF, or build with graphs ON and set
  `GGML_CUDA_FORCE_GRAPHS=1` for autoregressive workloads.
* **Cold-start (`~/.nv/ComputeCache` wiped) costs ~27 s on
  RTX 5090** when the host CUDA toolkit is older than the GPU
  architecture (toolkit 12.0 → emits PTX-89, driver JIT-compiles
  to sm_120 SASS at first dispatch). Driver-level disk cache
  amortises this across processes; no chatterbox-side patch can
  help. Mitigation: ship the build against CUDA Toolkit ≥ 12.8 so
  the binary contains native sm_120 SASS.
* **No size cap on the FP-error analysis.** For a different
  vocoder or a higher-precision down-stream consumer, the
  reduction-order change should be re-validated. A
  `test-cuda-ops` harness (matching `test-metal-ops`) would be
  the right place; not built today.
