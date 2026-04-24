# ggml patches for Chatterbox

`ggml` is vendored as a pristine upstream clone (see the top-level
[`README.md`](../README.md)), so any fixes we need in it live here as
standalone patches and are applied after the clone.

Two patches ship today:

1. [`ggml-metal-chatterbox-ops.patch`](#ggml-metal-chatterbox-opspatch) —
   fills gaps in the Metal backend (missing `diag_mask_inf`, front-pad
   `PAD`, scalar `conv_transpose_1d`, `MUL_MAT + ADD(+ADD)` fusion).
   Apple-only; harmless no-op on CPU / CUDA / Vulkan builds.
2. [`ggml-vulkan-pipeline-cache.patch`](#ggml-vulkan-pipeline-cachepatch)
   — persists the `VkPipelineCache` across processes so cold-start
   shader compile (seconds on fresh machines / containers / driver
   upgrades) drops to tens of ms.  Active whenever `-DGGML_VULKAN=ON`
   is in the build.

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

# Linux / Windows / Android — picks up the Vulkan pipeline-cache patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON

cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
```

If you'd rather run the steps by hand (e.g. to pin a different
upstream commit), the script is effectively:

```bash
git clone https://github.com/ggml-org/ggml.git ggml
cd ggml && git checkout $GGML_COMMIT
git apply ../patches/ggml-metal-chatterbox-ops.patch
git apply ../patches/ggml-vulkan-pipeline-cache.patch
```

`GGML_COMMIT` lives at the top of `scripts/setup-ggml.sh` as the
single source of truth — bump it when re-generating the patches
against a newer upstream ggml.  To confirm everything applied
cleanly:

```bash
(cd ggml && git status --short)
# Expected: 7 modified files under ggml/src/ggml-metal/
#           1 modified file under ggml/src/ggml-vulkan/
```

CPU-only or CUDA builds get the pinned commit but no useful patch
work: both targeted backends (Metal, Vulkan) are additive and
harmless when their compile flag is off.

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

No patch is needed for CPU / CUDA — those backends already handle
every op Chatterbox emits.

## `ggml-vulkan-pipeline-cache.patch`

Base commit: same as above (`58c3805`).

Adds an opt-in persistent `VkPipelineCache` to `ggml-vulkan`.  Upstream
ggml today calls `createComputePipeline(VK_NULL_HANDLE, …)`, i.e. no
pipeline cache at all, so every new process re-pays the driver's shader
compile wave (tens of ms to multiple seconds, dominated by the initial
`ggml_vk_load_shaders` call).  Drivers with aggressive system-wide
caches (recent NVIDIA, ANV) partially hide this; Mesa/RADV, Android
Adreno / Mali, fresh installs, and containers do not.

### What the patch does

1. Adds two fields to `vk_device_struct`:

   ```cpp
   vk::PipelineCache pipeline_cache = VK_NULL_HANDLE;
   std::string       pipeline_cache_path;
   ```

2. In `ggml_vk_get_device()` (right before `ggml_vk_load_shaders`), it
   resolves a cache directory from (in order)
   `$GGML_VK_PIPELINE_CACHE_DIR` → `$XDG_CACHE_HOME/ggml/vulkan` →
   `$HOME/.cache/ggml/vulkan`, `mkdir -p`'s it, loads any blob at
   `<dir>/<vendorID>-<deviceID>-<driverVersion>.pcache`, and calls
   `createPipelineCache(… seed …)`.  The feature is disabled by setting
   `GGML_VK_PIPELINE_CACHE_DIR=""`.

3. Changes the single `createComputePipeline(VK_NULL_HANDLE, …)` call
   to `createComputePipeline(device->pipeline_cache, …)`.  When caching
   is disabled the field is `VK_NULL_HANDLE` and behaviour is byte-
   identical to upstream.

4. Adds a helper `ggml_vk_save_pipeline_cache(vk_device &)` invoked
   from `ggml_vk_cleanup()` that dumps `getPipelineCacheData()` to the
   same path via atomic `<path>.tmp` + `rename`.  Doing the flush from
   `~vk_device_struct()` is unreliable — pipelines and helpers hold
   `shared_ptr<vk_device_struct>` refs that keep the refcount non-zero
   past typical process-exit, so the device destructor often never
   runs.

5. The destructor still calls `destroyPipelineCache` for resource
   cleanup in the event that refcounts do drop (e.g. a long-running
   server that tears down and re-creates backends).

Vulkan itself validates the pipeline-cache-header UUIDs / driver
version; if the on-disk blob becomes stale (driver bump, different
shader bundle, different GPU) it's silently ignored and pipelines are
recompiled into a fresh cache.  No manual invalidation needed.

### Measured impact (RTX 5090 + NVIDIA driver 590.48 + Vulkan 1.4)

Using `build/chatterbox` on the Turbo Q4_0 GGUFs, same prompt and seed.
"Cold" = target cache wiped via `rm -rf ~/.cache/{ggml,nvidia}`; "warm"
= left alone.  Each run is a fresh process:

| Scenario                                   | T3 infer | S3Gen infer | Wall   |
|--------------------------------------------|---------:|------------:|-------:|
| Both caches cold (fresh machine)           |  947 ms  |  1 741 ms   | 2 688 ms |
| ggml cache warm, driver cache cold         |   80 ms  |    166 ms   |   246 ms |
| Both caches warm (normal steady state)     |   69 ms  |    154 ms   |   223 ms |

Cold → ggml-warm saves **2.44 s** per process on fresh Linux — 91 % of
the full steady-state recovery.  On Android / Mesa / containers where
the driver cache doesn't help, this is the dominant wall-time cost and
the patch essentially eliminates it.  See
[`PROGRESS.md §3.11`](../PROGRESS.md) / the QVAC-17872 investigation
notes for the per-op profile that motivated this.

`build/test-*` binaries that don't call `ggml_backend_free` on exit
still get the cache seeded (from a prior process) but don't flush
updates on shutdown; this is fine — they only compile a subset of
pipelines each, and the main `chatterbox` / `chatterbox-tts` binaries
always write the superset.

### Caveats / follow-ups

* No opportunistic flush during long-running processes.  Shipping a
  second dispatcher that snapshots the cache every N minutes would
  make server-mode deployments resilient to crashes, but it's not
  needed for the CLI use cases today.
* We key the filename on `vendorID / deviceID / driverVersion`.  Two
  GPUs of the same model with different driver versions would reuse
  the same filename only when the driver matches, which is the
  intended sharing.
* No attempt to cap the on-disk size.  NVIDIA blobs land at ~1 MB for
  Turbo, ~2 MB for Turbo + MTL shared pipelines.  If this ever becomes
  a concern on mobile, wrap the write with a size check.
