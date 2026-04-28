# Changelog

All notable performance / correctness changes to chatterbox.cpp's
Vulkan path are tracked here, one entry per commit on the
`chatterbox-Optimize-cpp-backend-multilingual-model-for-Vulkan` branch.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
each section identifies the QVAC ticket the work belongs to (single
ticket QVAC-17872 for everything below).

The full per-round investigation lives alongside this file in
[`inputFilesForAI/qvac-17872-findings/FINDINGS*.md`](../inputFilesForAI/qvac-17872-findings/);
this file is the brief surface that ships with the repo.

---

## [Unreleased] — round-5: drop a wasted `ggml_cont` in `zero_pad_dim0`

**Status:** uncommitted (working tree only) at `src/chatterbox_tts.cpp`.

### Performance / cleanup

- `zero_pad_dim0()` previously wrapped the slice view in `ggml_cont`
  before passing to `ggml_scale(_, 0.0f)`.  That's wasted work — `cont`
  reads the source data into a fresh contiguous buffer and `scale`
  immediately overwrites it with zeros.  Vulkan's `ggml_scale`
  implementation handles strided sources via `nb[]` indexing, so we
  drop the `cont` and let `scale` read+zero in one pass.
- Removes ~30 `CONT` dispatches per CFM step on chatterbox-multilingual
  (one per padded `cfm_causal_block` / `cfm_causal_k3` / `zero_pad_dim0`
  call site).  Measured wall-clock delta on RTX 5090 / NVIDIA 590.48
  is **+0.10 ms (within the run-to-run noise floor, 2/6 pairwise wins)** —
  the optimization removes wasted GPU work but the per-cont cost
  was already trivial at these tensor sizes on this hardware.
- Real beneficiaries are the same cohort as round-2/3: bandwidth-starved
  Adreno / Mali / RADV / CPU-backend targets where the spurious read
  costs more than a per-stride increment in the scale shader.

### Correctness

- Bit-exact preserved across all three locked invariants:
  - Single-shot WAV `454b4cc14538e8ef917930b110d1e504`
  - Multi-synth identical-chunks PCM `4c83f367e6ca2b02fefbd480519ea3f6`
  - Multi-synth varied-length PCM `9252253ee532cb7928639a0f644a25da`

### Files changed

- `src/chatterbox_tts.cpp` — single function, ~6 line diff inside
  `zero_pad_dim0`.

---

## [Round-4] — 2026-04-28 — `cb5f883` "Round-4 ships a real desktop-GeForce optimization"

**Source:** `src/chatterbox_tts.cpp` (`+151 / -29`).

### Performance

- `basic_tfm` mid-block transformers in CFM previously did 3 separate
  `ggml_mul_mat` calls for Q / K / V.  The converter already coalesces
  these in `down_block` / `up_block` (the `m=652 n=256 k=768` shape
  in the perf log), but left the inner mid-block transformers as 3
  separate `(D, INNER)` matmuls — totalling **168 matmul dispatches /
  CFM step** (the dominant single shape on RTX 5090, 22 µs each).
- Round-4 fuses them into a single batched matmul via:
  1. concat the three weights along the **batch dim** (`dim=2`) into
     `W_qkv ne=(D, INNER, 3)`,
  2. broadcast `nx` to `ne[2]=3` via `ggml_repeat_4d` (sidesteps
     `ggml_can_mul_mat`'s `t1->ne[2] % t0->ne[2] == 0` constraint),
  3. one `ggml_mul_mat` → `ne=(INNER, T, 3)` with each batch a
     contiguous `(INNER, T)` slice — bit-identical to what 3 separate
     matmuls produce,
  4. plain `ggml_view_2d` per Q / K / V into the batched output, then
     the existing reshape→permute→cont chain for flash-attn.
- **Measured −4.58 ms / −3.3 % warm S3GEN_INFER on RTX 5090 / NVIDIA
  590.48** (n=6 alternating pairs; pairwise wins 6/6; ranges
  `[138.6, 141.2]` vs `[134.4, 136.5]`, non-overlapping).
- Default-on with env-var opt-out via `CHATTERBOX_FUSE_QKV=0` for the
  unlikely case a future driver / Vulkan version produces a different
  reduction order on the batched shape.

### Correctness

- Bit-exact preserved across all three locked invariants in BOTH
  default-on and opt-out paths.  Total: **85 individual MD5 / chunk
  checks**, every one matches.
- The opt-out branch is byte-identical to round-3 — literal `else`-arm
  of the same 3 `mul_mat` lines.

### Documented during landing

- Two layout pitfalls (concat-along-M not contiguous, mul_mat output
  layout — `(M, N)` with M innermost, not `(T, INNER)`) caught by the
  round-1 MD5 invariant before they shipped.  Full investigation:
  [`inputFilesForAI/qvac-17872-findings/FINDINGS_ROUND4.md`](../inputFilesForAI/qvac-17872-findings/FINDINGS_ROUND4.md).

---

## [Round-2 / 3] — 2026-04-28 — `a25e88f` "round 2 of optimizations"

**Source:**
- `src/chatterbox_tts.cpp` (+285 / −112)
- `patches/ggml-vulkan-eager-cache-save.patch` *(new)*
- `patches/README.md` (+88)
- `scripts/setup-ggml.sh` (+1)

This commit bundles two rounds of work; both keep `setup-ggml.sh`
idempotent on top of round-1.

### Round-2 — chatterbox-side stage-graph caches

- `s3gen_synthesize_to_wav`'s short-stage helpers (`compute_time_mlp`,
  `compute_time_mixed`, `run_encoder`, `run_f0_predictor`, `run_stft`,
  `run_hift_decode`) used to call `ggml_gallocr_new` /
  `ggml_gallocr_reserve` / `ggml_gallocr_alloc_graph` /
  `ggml_gallocr_free` per call (6 alloc cycles per Turbo synthesize,
  30 per MTL synthesize).  Each cycle pays Vulkan buffer alloc +
  descriptor-pool setup that mostly hurts bandwidth-starved mobile
  targets.
- New `stage_graph_cache_fixed` / `stage_graph_cache_keyed` /
  `hift_graph_cache` structs hold the `ggml_context` / `ggml_cgraph`
  / `ggml_gallocr_t` / host scratch buffer alive across calls,
  rebuilding only when the cache key (`T` / `T_mel` / `T_stft`)
  changes.  Lifetime is bound to the s3gen backend — both
  `s3gen_model_cache_release` (atexit) and the `s3gen_model_cache_get`
  cache-miss path drop the stage caches BEFORE freeing the backend,
  so we never `gallocr_free` against a dangling `vk_device`.
- Bit-exact preserved on round-1's single-shot WAV invariant
  `454b4cc14538e8ef917930b110d1e504`.  New round-2 invariant locked
  for multi-synth identical chunks: `4c83f367e6ca2b02fefbd480519ea3f6`.

### Round-2 — `patches/ggml-vulkan-eager-cache-save.patch`

- Round-1's `ggml-vulkan-pipeline-cache.patch` writes the on-disk
  cache only on backend-free (`ggml_vk_cleanup`).  Chatterbox
  compiles all shaders **lazily** during the first graph compute; if
  the process crashes (OOM, signal, abort) before cleanup, all the
  compiled pipelines are lost.
- The new patch persists the cache eagerly from
  `ggml_vk_load_shaders` whenever the call actually grew the cache
  (size-tracked via a new `pipeline_cache_last_size` field on
  `vk_device_struct`).  Warm runs where every pipeline is a cache
  hit (size unchanged) skip the disk write at zero overhead — the
  naive `!compiles.empty()` guard regressed warm WALL by **+90 ms**
  by re-writing the same 1 MB blob ~60 times per inference; that
  was caught by the round-1 invariant during this round.
- Stacks cleanly on round-1's pipeline-cache patch.  Idempotent
  setup-ggml.sh re-run.

### Round-3 — skip constant-input re-upload on cache hit

*Bundled into the same commit as round-2.*

- Each stage cache now owns a `bool inputs_uploaded` flag plus the
  CPU-side constant-valued inputs that go with the cached graph
  (`pe1_data` + `pe2_data` for encoder, `kernel_data` for STFT,
  `istft_k_data` + `w_sum_data` + existing `inv_alphas` for HiFT).
  On cache-hit calls, `compute_pos_emb` / `build_hann_window` /
  `build_istft_kernel` / `build_window_sum` / `invert_alpha_cpu`
  no longer re-run, and the same data is no longer re-uploaded
  to the GPU.
- `inputs_uploaded` is reset by `destroy()` so any cache-rebuild
  (different `T`, or `s3gen_model_cache_release` at process exit)
  goes through the fresh-upload path automatically.
- New round-3 invariant locked for multi-synth varied-length:
  `9252253ee532cb7928639a0f644a25da`.  This is the only invariant
  that catches a `destroy()` that fails to clear `inputs_uploaded`.

### Performance (rounds 2+3 combined, RTX 5090 alternating)

| Metric (warm S3GEN_INFER) | Round-1 baseline | After round-2/3 | Δ                |
|---------------------------|-----------------:|----------------:|-----------------:|
| Mean (n=3 alt pairs)      |        130.4 ms  |       129.1 ms  |  −1.3 ms / −1.0 % |

The single-shot CLI on RTX 5090 was already past the regime where CPU
preprocessing or host→device transfers move the needle; the rounds-2/3
wins are felt in multi-call streaming and on bandwidth-starved targets.

### Investigation

- [`inputFilesForAI/qvac-17872-findings/FINDINGS_ROUND2.md`](../inputFilesForAI/qvac-17872-findings/FINDINGS_ROUND2.md)
- [`inputFilesForAI/qvac-17872-findings/FINDINGS_ROUND3.md`](../inputFilesForAI/qvac-17872-findings/FINDINGS_ROUND3.md)
- `bench-logs-vk-round2/` and `bench-logs-vk-round3/`

---

## [Round-1] — 2026-04-24 — `49b1a32` "QVAC-17872 [TTS GGML] Optimize cpp backend multilingual model for Vulkan"

**Source:**
- `patches/ggml-vulkan-pipeline-cache.patch` *(new)*
- `patches/README.md` (+147 / −0; new Vulkan section)
- `scripts/setup-ggml.sh` (+45 / −13; PATCHES array, idempotent)

### Added

- Persistent `VkPipelineCache` patch for the vendored ggml
  (`ggml-vulkan-pipeline-cache.patch`).  Adds a `vk::PipelineCache`
  field on `vk_device_struct` seeded at init from
  `$GGML_VK_PIPELINE_CACHE_DIR` / `$XDG_CACHE_HOME/ggml/vulkan` /
  `$HOME/.cache/ggml/vulkan` keyed on
  `vendorID-deviceID-driverVersion`, threaded through every
  `createComputePipeline` call, and flushed back to disk from
  `ggml_vk_cleanup` (NOT from `~vk_device_struct` because pipelines
  hold `shared_ptr<vk_device_struct>` refs that keep the refcount
  above zero past process exit — the destructor often never runs).
- Atomic `<path>.tmp + rename` writes; opt-out via
  `GGML_VK_PIPELINE_CACHE_DIR=""`; Vulkan's own header validation
  silently invalidates stale blobs so no manual cache-busting needed.

### Performance — RTX 5090 / NVIDIA 590.48 / Vulkan 1.4.325, fresh process each

| Scenario                                 | T3       | S3Gen    | Wall    |
|------------------------------------------|---------:|---------:|--------:|
| Both caches cold (fresh machine)         |   947 ms | 1 741 ms | 2 688 ms |
| ggml cache warm, NVIDIA driver cache cold|    80 ms |   166 ms |   246 ms |
| Both caches warm (steady state)          |    69 ms |   154 ms |   223 ms |

The middle row is the one round-1 adds: **−2.44 s recovered per
process** on a cold driver cache (91 % of the cold→warm gap).  On
Linux/NVIDIA with a warm driver cache it still saves ~10–30 ms of
first-graph dispatch; on Mesa / Adreno / Mali / containers — where
the driver cache doesn't help — it eliminates the full seconds.

### Investigation

- [`inputFilesForAI/qvac-17872-findings/FINDINGS.md`](../inputFilesForAI/qvac-17872-findings/FINDINGS.md)
  — the original Vulkan profiling pass; identifies the §5.1
  pipeline-cache opportunity and §5.2-§5.7 follow-ups (most of
  which became rounds 2–4).
- [`inputFilesForAI/qvac-17872-findings/PR_DESCRIPTION.md`](../inputFilesForAI/qvac-17872-findings/PR_DESCRIPTION.md)
- `bench-logs-vk-round2/baseline-*.log` (locks the round-1 single-shot
  invariant `454b4cc14538e8ef917930b110d1e504`).

---

## Conventions

* **Bit-exact invariants:** every commit on this branch must pass
  the locked single-shot WAV md5
  `454b4cc14538e8ef917930b110d1e504`.  Rounds 2 + 3 add
  multi-synth invariants verifying cache-hit and cache-rebuild paths.
  Run `bench-logs-vk-round3/regress-tight.sh build-* round-N` and
  `bench-logs-vk-round3/regress-multi-varied.sh build-* round-N`
  before pushing.
* **Perf measurement:** alternating `regress-tight.sh` (1 iter, 16
  chunks each) between baseline and target, for thermal-controlled
  comparison; aggregate over n≥6 pairs.  Single-iter runs at the
  same thermal state are tighter than 5-iter aggregates that span
  drift.
* **Opt-out env-vars introduced** (all default to the new
  fast-path; explicit `=0` reverts to the prior behaviour, identical
  to the previous round):
  - Round-1: `GGML_VK_PIPELINE_CACHE_DIR=""` — disables on-disk
    pipeline cache entirely.
  - Round-4: `CHATTERBOX_FUSE_QKV=0` — disables Q/K/V matmul fusion
    in CFM `basic_tfm` mid-blocks.
