# Changelog

All notable changes on the
`chatterbox-Optimize-cpp-backend-multilingual-model-for-CUDA` branch
(QVAC-17873, the CUDA companion to QVAC-17872 / Vulkan).

The branch is a strict superset of `main` — every change is additive
(new patch + new tests + build-system extension), no chatterbox C++
source is touched.  All four commits stack onto each other and the
current uncommitted work continues from the last commit.

Each commit is also a self-contained "stage" of the investigation
recorded in `inputFilesForAI/qvac-17872-findings/FINDINGS_CUDA.md`;
see § 5.x cross-references below for the full diagnostic / measurement
context.

Format follows [Keep a Changelog](https://keepachangelog.com/),
adapted for branch-level vendored-patch development (no semver
because this is upstream-targeted patch work, not a release).

---

## [Unreleased]

Work-in-progress on top of the round-6 commit (3-op fusion).
Two parts so far: Part 5 (round 7) shipped the `GGML_CUDA_FATTN_KERNEL`
override + variant sweep; Part 6 (round 8, this work) closes the
investigation by validating production stability + diversity and
documenting the remaining flash-attn gap as a Blackwell config-table
issue best fixed upstream.

### Part 6 — Production stability + diversity validation (round 8)

#### Added
- **`scripts/test-stability.sh`** — 50-run sequential soak test.
  Asserts: (1) every run is **bit-identical** (same wav md5, no
  process-to-process state leaks), (2) median wall-time drift between
  warm-up window (runs 5-9) and steady-state (runs 46-50) is **≤ ±5 %**
  for both T3 and S3Gen (catches per-run cache growth, GPU memory
  fragmentation, pool exhaustion), (3) RSS growth from warm-up to
  steady-state is **≤ 10 MB** (catches process-level memory leaks,
  with a slack for libc heap stabilisation), (4) `~/.nv/ComputeCache/`
  growth is **≤ 50 MB** (catches accidental JIT recompiles that
  shouldn't happen on a stable native-SASS build).  Scales gracefully
  to small N for smoke testing — for N < 20 it picks first-half /
  second-half windows automatically.  Result on the post-round-7 build:
  **all 50 runs identical (md5 `fab3a8f4…`), 0 % drift on T3 and
  S3Gen, RSS −152 KB (slight shrink, no leak), 0 KB JIT-cache
  delta**.  Bench log:
  `inputFilesForAI/qvac-17872-findings/bench-logs-cuda/stability-soak50.log`.

- **`scripts/test-diversity.sh`** — edge-case sweep (no bit-identity
  asserted across different inputs, just exit-zero + non-empty-wav +
  reasonable size).  Phase 1: 5 distinct seeds (0, 1, 7, 42,
  1000003) — asserts ≥ 2 distinct outputs (sampler is seed-responsive).
  Phase 2: 5 multilingual prompts (EN, FR, ES, DE, IT) — asserts all 5
  outputs distinct (multilingual BPE path works).  Phase 3: 3
  edge-case durations (`"Hi."` ≈ 2 tokens, ~50-word sentence,
  whitespace-padded `"  word   "`).  Result on post-round-7 build:
  **all 13 cases produce non-empty distinct wavs, no crashes; T3
  ranges 66 ms (`"Hi."`) → 613 ms (long prompt)**.  Bench log:
  `inputFilesForAI/qvac-17872-findings/bench-logs-cuda/diversity-test.log`.

#### Investigation finding (no patch)
- **Root-cause of the remaining 67 ms / utterance flash-attn gap to
  Vulkan:** code-inspection of `ggml/src/ggml-cuda/fattn-mma-f16.cuh`
  shows that the per-architecture MMA configuration table
  (`ggml_cuda_fattn_mma_get_config_<arch>`) has **no Blackwell entry**.
  `ggml_cuda_fattn_mma_get_config(cc)` checks
  `ampere_mma_available(cc)` first, which returns `true` for any
  `cc >= GGML_CUDA_CC_AMPERE (800)` — so RTX 5090 (sm_120) silently
  uses the **Ampere config tuned for sm_80**:
  `nthreads=128, occupancy=2, nbatch_fa=64, nbatch_K2=32,
  nbatch_V2=32, nstages=2, Q_in_reg=true` for our (DKQ=64, DV=64,
  ncols=64) shape.  Blackwell SMs have larger register files and
  more shared memory than sm_80, so this config is almost
  certainly leaving perf on the table.  Tuning a Blackwell config
  empirically requires either NVIDIA Nsight Compute hardware
  counters (blocked on this host without root-level
  `NVreg_RestrictProfilingToAdminUsers=0`) or 5-10 multi-hour
  `nthreads × occupancy × nbatch_fa × nbatch_KV` sweeps with full
  rebuild per point.  Best deferred to upstream maintainers who
  have access to multiple Blackwell SKUs and can land a config-table
  entry that benefits everyone.  The `GGML_CUDA_FATTN_KERNEL` env
  var shipped in Part 5 lets them confirm/refute on their own
  hardware quickly (it surfaces the picker decision and lets them
  A/B variants without rebuilding).  See `FINDINGS_CUDA.md` § 5.6
  for the full code-trace and decision rationale.

#### Test harness update
- Validation harness expands from **5 → 7** test artefacts and
  **47 → 47 + 7 = 54** assertions covering kernel correctness,
  build-system idempotency, end-to-end audio bit-identity, perf-logger
  output format, FlashAttention variant safety, **50-run production
  soak**, and **multilingual / multi-seed / edge-duration coverage**.
  Detailed inventory in `patches/README.md` "Validation harness".

---

### Part 5 — `GGML_CUDA_FATTN_KERNEL` override (round 7)

#### Added
- **`GGML_CUDA_FATTN_KERNEL=tile|mma|wmma|vec` opt-in env var** that
  overrides the FlashAttention picker on calls where the default
  heuristic chose `MMA_F16` (large-batch / prompt-phase path).
  Mirrors the diagnostic capability in `GGML_VK_*` env vars on
  ggml-vulkan.  Per-arch availability (compiled-in SASS check) and
  per-shape compatibility (`Q.ne[1]`, `K.ne[1] %
  FATTN_KQ_STRIDE`) are validated; unsupported overrides fall back
  to the default picker with a one-shot `GGML_LOG_WARN` — never
  trip the dispatcher's `GGML_ABORT`.  ~140-line addition to
  `ggml/src/ggml-cuda/fattn.cu` (4 helpers + a wrapper around the
  existing picker, which is renamed to `_default`).  See
  `patches/README.md` Part 5.
- **Empirical finding** from the variant sweep on RTX 5090 + Turbo
  Q4_0: `default` and overrides to `mma` / `wmma` / `vec` are all
  bit-identical (`tile` is the only one that actually changes
  kernel choice — and it's 4 % slower than MMA).  Conclusion:
  picker is already optimal for chatterbox on Blackwell; the
  remaining 67 ms / utterance flash-attn gap to Vulkan is
  intrinsic to MMA_F16's kernel quality, not a picker selection
  issue.  Out-of-scope kernel rewrite for this round.
- 5 new flash-attn correctness cases in `src/test_cuda_ops.cpp`
  (step n_q=1, step n_kv=383, prompt 383 masked, prompt 64
  masked, tiny).  NMSE ≤ 5e-4 (matches ggml's
  `test-backend-ops` threshold for fp16-accumulate ops).
- `scripts/bench-fattn-variants.sh` — 3-phase variant sweep
  benchmark.  Phase 1 measures T3 / S3Gen wall time per variant
  (median of N runs), phase 2 collects per-op `FLASH_ATTN_EXT`
  total via `GGML_CUDA_PERF_LOGGER=1`, phase 3 reports a ranking
  + audio bit-identity vs the default-picker output.  Safe to run
  on unpatched builds (env var is ignored, all variants
  bit-identical).  Saved bench log:
  `inputFilesForAI/qvac-17872-findings/bench-logs-cuda/fattn-variants-bench.log`.

### Investigation log (per the user's "do code audit every time" directive)

Each implementation pass was preceded by an audit:

1. **Pre-implementation audit** of `fattn.cu` revealed all 4
   variants are technically dispatchable for chatterbox shapes
   (head_dim=64, F16 K/V); minimum-surface design is renaming
   the existing picker to `_default` and wrapping it.
2. **Mid-implementation audit (1)**: forcing WMMA on Blackwell
   crashes with *"CUDA kernel has no device code compatible with
   CUDA arch 1200"* — added `ggml_cuda_fattn_variant_available`
   to mirror the picker's existing arch gates.
3. **Mid-implementation audit (2)**: forcing VEC on chatterbox's
   prompt-phase shape crashes with *"CUDA error: invalid
   configuration argument"* — VEC's compile-time templates only
   support `ncols ≤ 2`.  Added scope narrowing
   (`def == MMA_F16`-only override) and a per-shape gate that
   mirrors `can_use_vector_kernel` from the picker.
4. **Mid-implementation audit (3)**: VEC override silently
   crashed T3 (1001 garbage tokens instead of 18) — debug print
   showed default picker actually selects MMA_F16 for step phase
   too because `K.ne[1] % FATTN_KQ_STRIDE != 0` (KV cache grows
   by 1 per token, FATTN_KQ_STRIDE=256).  Per-shape gate
   strengthened to include the `K.ne[1]` divisibility check.
5. **Final audit / regression**: 47 distinct PASS assertions
   across 5 test suites, clean re-setup builds + binary
   verifies on every variant override.

---

## [446f94b] — 2026-04-27 11:10 — Part 4: `GGML_CUDA_PERF_LOGGER=1` shipped + tested

Adds a CUDA per-op timing logger that mirrors `ggml-vulkan`'s
`GGML_VK_PERF_LOGGER=1`.  See `FINDINGS_CUDA.md` § 5.1.c for the
investigation context.

### Added
- `GGML_CUDA_PERF_LOGGER=1` env var.  When set, the dispatch loop
  in `ggml_backend_cuda_graph_compute` wraps each per-op section
  with `cudaEventRecord(start) / cudaEventRecord(end)` via an RAII
  `scope` helper (fusion fast-paths included — destructor fires on
  every `continue`).  After every `ggml_backend_cuda_graph_compute`
  call, the logger `cudaStreamSynchronize`s, reads
  `cudaEventElapsedTime` for every recorded pair, aggregates by
  `(op-kind, dtype, shape)` key, and prints to `stderr`.
- Output format matches Vulkan's `vk_perf_logger`:
  ```
  ----------------
  CUDA Timings:
  MUL_MAT q4_0 m=3072 n=383 k=1024: 24 x 241.979 us = 5807.507 us
  …
  Total time: 22480.220 us.
  ```
  Cross-backend grep / awk one-liners (the reproduction recipes in
  `FINDINGS.md` § 7) work unchanged for both backends.
- `scripts/test-cuda-perf-logger.sh` — 4-phase end-to-end test:
  (1) default run produces no perf-logger output and audio is
  identical to a control run; (2) env-var run produces ≥ 1
  `CUDA Timings:` block per `compute_graph` call, line format is
  `<name>: N x A.B us = T.U us`, hot ops MUL_MAT / FLASH_ATTN_EXT /
  NORM / CONV_TRANSPOSE_1D all present; (3) `PERF_LOGGER=1` +
  `FORCE_GRAPHS=1` together still produces output (proves graphs
  auto-disable when perf logger is on); (4) aggregate-time sanity
  bound (last `Total time:` in [1 µs, 1 s]).

### Patch contents (`patches/ggml-cuda-chatterbox-ops.patch`, +323 lines)
- `ggml/src/ggml-cuda/ggml-cuda.cu`: ~280-line `ggml_cuda_perf_logger`
  class (Meyers singleton, RAII `scope` helper, `cudaEvent_t` pool
  with on-demand growth, aggregation map, sorted print).  Per-op
  scope guard added in the dispatch loop in
  `ggml_cuda_graph_evaluate_and_capture`.  `flush_and_print` hook
  added at the end of `ggml_backend_cuda_graph_compute`.
- `ggml/src/ggml-cuda/common.cuh`: `ggml_cuda_graph::is_enabled()`
  extended with `disable_cuda_graphs_due_to_perf_logger` so graphs
  auto-disable when `GGML_CUDA_PERF_LOGGER=1`.

### Why graphs auto-disable when the logger is on
CUDA Graph capture would either hide individual-op timings inside
`cudaGraphLaunch` (event values not visible until launch fires) or
re-record over still-pending events on subsequent launches.  The
logger is diagnostic-only and graphs are off by default for
chatterbox anyway (FINDINGS_CUDA.md § 3.4), so the simplest correct
behaviour is mutual exclusion.

### Code-audit fixes during development
- `cudaEventDestroy` removed from the singleton's destructor —
  Meyers-singleton dtor runs at static destruction time (after main
  returns and often after libcudart's own statics), where calling
  `cudaEventDestroy` can crash on torn-down driver.  Events are
  leaked to OS process-exit reclaim; the pool is bounded
  (~8 K events).
- `cudaEventCreate` failure (e.g. OOM) tolerated: failed slots are
  marked `nullptr`, recording skips them, `flush_and_print` filters
  them out before reading elapsed time.
- Aggregator cleared after every print (matches Vulkan's
  `vk_perf_logger::print_timings`).  Cross-call cumulation would
  mix prompt-phase (n=large) and step-phase (n=1) MUL_MAT rows,
  hiding the shape distribution.

### Files changed
| Path                                       | Δ      |
|--------------------------------------------|-------:|
| `patches/README.md`                        | +122   |
| `patches/ggml-cuda-chatterbox-ops.patch`   | +323   |
| `scripts/test-cuda-perf-logger.sh` (new)   | +184   |

---

## [60e4233] — 2026-04-27 10:08 — Tests added (kernel correctness + end-to-end smoke)

Drops two regression tests in place so future patch revisions
(rebases against newer ggml, follow-up perf fixes) can be validated
in ~1 minute without manual benchmarking.

### Added
- `src/test_cuda_ops.cpp` — kernel-level CPU vs CUDA validation
  for the patched `conv_transpose_1d` (counterpart to the existing
  `test-metal-ops` for the Metal patch).  13 cases covering
  HiFT-realistic shapes (3 layer sizes), warp-reduction edge cases
  (`IC < warp / IC = warp / IC > warp`), stride/kernel edge cases
  (`K==s0`, `s0==1`, `K > IL`), and tiny shapes (1×1, 2×2).
  Tolerance `1e-3` abs / `1e-3` rel; observed worst case is `7.2e-7`
  on HiFT shapes.  No-op when CUDA isn't enabled — exits 0 with a
  notice, so safe to wire into non-CUDA CI.
- `scripts/test-chatterbox-cuda.sh` — end-to-end smoke test, three
  phases: (1) `FORCE_GRAPHS=1` produces bit-identical audio to
  default (verified via `cmp -s` on the wav); (2) 18-run stress
  matrix (3 seeds × 3 prompt lengths × 2 modes), check no crash
  and no empty output; (3) perf sanity (`FORCE_GRAPHS` doesn't
  regress T3 by more than 5 % vs default).
- `CMakeLists.txt`: new `test-cuda-ops` build target.

### Files changed
| Path                                  | Δ     |
|---------------------------------------|------:|
| `CMakeLists.txt`                      |  +10  |
| `patches/README.md`                   |  +22  |
| `scripts/test-chatterbox-cuda.sh` (new) | +141  |
| `src/test_cuda_ops.cpp` (new)         | +159  |

---

## [9d2fb77] — 2026-04-27 09:51 — Part 2: `GGML_CUDA_FORCE_GRAPHS=1`

Adds an opt-in env var that bypasses the strict 2-call warmup check
in `ggml_backend_cuda_graph_compute` so chatterbox-style autoregressive
workloads (T3 step decode with growing K/V views) actually hit the
captured-graph cache.  See `FINDINGS_CUDA.md` § 5.1.b.

### Background
ggml-cuda's default warmup logic requires every property of every
node to be byte-identical between the first two calls before
`warmup_complete = true`.  Right default for llama.cpp-style decode
where each call sends a stable cgraph with stable data pointers;
**wrong default for chatterbox T3** which builds a fresh-but-
topologically-identical cgraph per token *with growing K/V views*
(`L = n_past + 1`).  Each call therefore looks "different" to the
strict check (K's `ne[1]` grows by 1, view offsets shift) so
`warmup_complete` keeps resetting and graphs are never used.  Cost:
the per-token kernel-launch overhead is visible as the ~90 ms gap
between T3 GPU work (~70 ms) and T3 wall (163 ms) on RTX 5090.

### What the env var does
When `GGML_CUDA_FORCE_GRAPHS=1` is set, the compute path always uses
graphs and trusts the existing `cudaGraphExecUpdate` /
re-instantiate-on-failure wiring in
`ggml_cuda_graph_update_executable` to absorb per-call differences:

```cpp
if (force_graphs) {
    use_cuda_graph = true;
    cuda_graph_update_required = properties_changed || graph->instance == nullptr;
}
```

Default behaviour (env var unset) is unchanged.

### Measured impact (RTX 5090, Turbo Q4_0, fresh-process median)

| Prompt tokens | Default T3 | `FORCE_GRAPHS=1` T3 | Δ ms   | Δ %   |
|--------------:|-----------:|--------------------:|-------:|------:|
|           19  |     120 ms |            113 ms   | -7 ms  | -6 %  |
|           43  |     163 ms |            150 ms   | -13 ms | -8 %  |
|          157  |     384 ms |            332 ms   | -52 ms | -14 % |
|          231  |     523 ms |            453 ms   | -70 ms | -13 % |

Per-token saving is constant ~0.3 ms/token across prompt lengths.
Audio output is **bit-identical** with vs without the env var
(verified by `md5sum` on 43- and 231-token runs).

### Files changed
| Path                                       | Δ      |
|--------------------------------------------|-------:|
| `patches/README.md`                        | +122   |
| `patches/ggml-cuda-chatterbox-ops.patch`   |  +42   |

---

## [e2483ac] — 2026-04-27 09:24 — QVAC-17873 part 1: warp-cooperative `conv_transpose_1d` kernel

Initial work item for the QVAC-17873 ticket.  Adds a CUDA-side
patch + build-system support modelled after the existing Metal
patch.  See `FINDINGS_CUDA.md` § 5.1.

### Problem
ggml-cuda's `conv_transpose_1d_kernel` is a textbook scalar
implementation: one CUDA thread per output pixel, scanning all
`IC × IL` input positions with an `if (idx >= i*s0 && idx < i*s0+K)
continue;` skip conditional.  For HiFT upsample shapes (K=16, s0=8,
IL≈100, IC≈512) only ~`K/s0 = 2` of the IL iterations contribute;
the rest are pure dispatch waste.

`nsys` measured this as **#1 GPU kernel by a wide margin**: 4 calls
totalling 135.98 ms (largest single call: 101.34 ms), 67 % of
total S3Gen GPU time.  ggml-vulkan ran the same shapes in
3.42 ms / 4 calls — **40× faster**.  Same shape problem the Metal
patch's `kernel_conv_transpose_1d` rewrite fixed for Apple
(PROGRESS.md § 3.12).

### Fix
Replaces the kernel with a **warp-cooperative variant** modelled on
the Metal-patch design (one threadgroup per output pixel + simdgroup
reduction across input channels):

1. **Grid `(OL, OC, 1)` × block `(32, 1, 1)`** — one CUDA warp per
   output pixel.
2. **Analytical `i_start, i_end`** clamping from
   `i*s0 + ki = ol, 0 ≤ ki < K, 0 ≤ i < IL` so the inner `i`
   loop only iterates over the contributing range (≤ `K/s0 + 1`
   positions).  The skip conditional is eliminated.
3. **IC reduction parallelised across the warp** — each thread
   handles a strided slice `ic = tid, tid+32, …` and accumulates
   a partial sum.
4. **Warp reduction** via `__shfl_xor_sync(0xFFFFFFFFu, v, …)`,
   thread 0 writes the final value to `dst`.

Block-size constant `CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE` drops from
256 → 32 to match.  Host-side `ggml_cuda_op_conv_transpose_1d` entry
point and shape constraints (contiguous src0/src1, ne1==ne3==1) are
unchanged.

### Measured impact (RTX 5090, Turbo Q4_0, warm)

| Stage / kernel                          | Stock      | Patched   | Speedup |
|-----------------------------------------|-----------:|----------:|--------:|
| `conv_transpose_1d_kernel` (4 inst., GPU) | 135.98 ms | 3.21 ms | **42×** |
| `[hift_decode]`                         |  149.5 ms  |  22.0 ms  | 6.8×    |
| `[hift_total]`                          |  144.7 ms  |  30.0 ms  | 4.7×    |
| `S3GEN_INFER_MS`                        |   280 ms   |  170 ms   | 1.6×    |
| Wall (T3 + S3Gen)                       |   443 ms   |  333 ms   | 1.3×    |
| RTF                                     |   0.15     |  0.09     |   —     |

### Build-system integration
- `scripts/setup-ggml.sh`: switched to a `PATCHES=(…)` array,
  applies `ggml-metal-chatterbox-ops.patch` and the new
  `ggml-cuda-chatterbox-ops.patch` in order.  Idempotent via
  per-patch `git apply --check` (later refined to `--reverse
  --check` in the Unreleased section).  Banner mentions
  `-DGGML_CUDA=ON` alongside the Metal flag.
- `patches/README.md`: documents the new patch alongside the Metal
  one, including the trade-off discussion (FP-reduction-order
  change introduces -58 dBFS / SNR 58.5 dB delta vs the stock
  kernel — inaudible, same kind of FP-order variance the Metal
  patch's `simd_sum` introduces).

### Correctness
End-to-end audio comparison on the same seed + prompt:

| Metric                | Value             |
|-----------------------|-------------------|
| Sample count          | 44 160 (same)     |
| Max abs sample diff   | 0.001221 (-58 dBFS) |
| RMS error             | 0.000043          |
| SNR (signal vs diff)  | **58.5 dB**       |

Inaudible.  `git apply --check
patches/ggml-cuda-chatterbox-ops.patch` is clean against vanilla
ggml@58c38058.

### Files changed
| Path                                          | Δ       |
|-----------------------------------------------|--------:|
| `patches/README.md`                           |  +189   |
| `patches/ggml-cuda-chatterbox-ops.patch` (new)|  +162   |
| `scripts/setup-ggml.sh`                       |   +45   |

---

## Cumulative impact across the four commits

End-to-end on RTX 5090 + Turbo Q4_0 (Hello from ggml., warm,
median of 5 fresh-process runs):

| Stage                   | main        | Branch HEAD     | Δ        |
|-------------------------|------------:|----------------:|---------:|
| `[hift_decode]`         |   149.5 ms  |    22.0 ms      | -85 %    |
| `[hift_total]`          |   144.7 ms  |    30.0 ms      | -79 %    |
| `S3GEN_INFER_MS`        |   280 ms    |   170 ms        | -39 %    |
| T3 (default)            |   163 ms    |   163 ms        |   0 %    |
| T3 (`FORCE_GRAPHS=1`)   |   163 ms    |   150 ms        |  -8 %    |
| Wall (T3 + S3Gen)       |   443 ms    |   333 ms        | -25 %    |
| RTF                     |   0.15      |   0.09          | -40 %    |

Stacking with the externally-validated **CUDA Toolkit 12.0 → 12.8
upgrade** (no source patch — see FINDINGS_CUDA.md § 5.2 / § 2b):

| Metric (CUDA 12.8 + branch HEAD + FORCE_GRAPHS) | Value                |
|-------------------------------------------------|----------------------|
| **Cold start** (`~/.nv/ComputeCache` wiped)     | **1 102 ms** (was 26 788 ms — **24×**) |
| Warm T3                                         | 97 ms                |
| Per-token T3 (long prompt)                      | 1.73 ms (was 2.27)   |
| Distance to Vulkan parity (long prompt)         | 1.4× (was 2.6×)      |

## Test coverage on branch HEAD + Unreleased

35 distinct assertions, full suite runs ~2 minutes on RTX 5090:

| Test                               | Assertions | Catches                                              |
|------------------------------------|-----------:|------------------------------------------------------|
| `build/test-cuda-ops`              |        13  | conv_transpose_1d kernel correctness vs CPU          |
| `scripts/test-build-system.sh`     |         4  | setup-ggml.sh idempotency / corruption recovery / patch-apply |
| `scripts/test-chatterbox-cuda.sh`  |        14  | end-to-end + bit-identity + env-var matrix           |
| `scripts/test-cuda-perf-logger.sh` |         4  | perf-logger format / graph-disable / sane time       |

Run order: 2 → build → 1 → 3 → 4.

## Reproducibility

```bash
# 1. Get nvcc 12.8+ (CUDA 12.0 + Ubuntu 24.04 has a glibc/_Float32
#    compatibility issue unrelated to this work).
sudo apt install -y cuda-toolkit-12-8
export PATH=/usr/local/cuda-12.8/bin:$PATH

# 2. Build chatterbox + tests.
./scripts/setup-ggml.sh                                # applies all patches
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=$(which nvcc)
cmake --build build-cuda -j$(nproc) --target chatterbox test-cuda-ops

# 3. Get the Turbo Q4_0 GGUFs once.
python3 scripts/convert-t3-turbo-to-gguf.py --quant q4_0 \
    --out models/chatterbox-t3-turbo-q4_0.gguf
python3 scripts/convert-s3gen-to-gguf.py \
    --out models/chatterbox-s3gen-turbo.gguf

# 4. Run the full regression sweep.
./build-cuda/test-cuda-ops
./scripts/test-build-system.sh
./scripts/test-chatterbox-cuda.sh
./scripts/test-cuda-perf-logger.sh
```

All four expected to print `*** PASSED` / `All * tests PASSED.`.
