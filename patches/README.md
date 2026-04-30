# ggml patches for Chatterbox

`ggml` is vendored as a pristine upstream clone (see the top-level
[`README.md`](../README.md)), so any fixes we need in it live here as
standalone patches and are applied after the clone.

Three patches ship today:

1. [`ggml-metal-chatterbox-ops.patch`](#ggml-metal-chatterbox-opspatch) —
   fills gaps in the Metal backend (missing `diag_mask_inf`, front-pad
   `PAD`, scalar `conv_transpose_1d`, `MUL_MAT + ADD(+ADD)` fusion).
   Apple-only; harmless no-op on CPU / CUDA / OpenCL / Vulkan builds.
2. [`ggml-opencl-chatterbox-ops.patch`](#ggml-opencl-chatterbox-opspatch) —
   fills gaps in the OpenCL backend (`CONV_TRANSPOSE_1D` for HiFT, `SIN`,
   backend-init nullability docs).  Active when `-DGGML_OPENCL=ON`;
   inert on builds that don't link OpenCL.
3. [`ggml-vulkan-pipeline-cache.patch`](#ggml-vulkan-pipeline-cachepatch)
   — persists the `VkPipelineCache` across processes so cold-start
   shader compile (seconds on fresh machines / containers / driver
   upgrades) drops to tens of ms.  Active whenever `-DGGML_VULKAN=ON`
   is in the build.
3. [`ggml-vulkan-eager-cache-save.patch`](#ggml-vulkan-eager-cache-savepatch)
   *(QVAC-17872 round-2)* — extends patch 2 with size-tracked eager
   flushes from `ggml_vk_load_shaders`, so a process crash mid-cold-
   compile no longer discards the lazy-compile work.  Stacks on top of
   patch 2 (depends on the `pipeline_cache` field it adds).

`scripts/setup-ggml.sh` applies all three in order; the patches stack
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

# Android / Termux — picks up the OpenCL patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENCL=ON

# Linux / Windows / Android — picks up the Vulkan pipeline-cache patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON

cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
```

If you'd rather run the steps by hand (e.g. to pin a different
upstream commit), the script is effectively:

```bash
git clone https://github.com/ggml-org/ggml.git ggml
cd ggml && git reset --hard $GGML_COMMIT && git clean -fdq
git apply ../patches/ggml-metal-chatterbox-ops.patch
git apply ../patches/ggml-opencl-chatterbox-ops.patch
git apply ../patches/ggml-vulkan-pipeline-cache.patch
```

`GGML_COMMIT` lives at the top of `scripts/setup-ggml.sh` as the
single source of truth — bump it when re-generating the patches
against a newer upstream ggml.  To confirm everything applied
cleanly:

```bash
(cd ggml && git status --short)
# Expected: ~7 modified files under src/ggml-metal/
#           include/ggml-opencl.h + a handful under src/ggml-opencl/
#           1 modified file under src/ggml-vulkan/
```

Skip `setup-ggml.sh` only if you use `-DTTS_CPP_USE_SYSTEM_GGML=ON`
with another ggml; otherwise the pin + patches keep builds
deterministic.

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

## `ggml-opencl-chatterbox-ops.patch`

Base commit: `58c3805` (same pin as the Metal patch).

Extends `ggml-opencl` so Chatterbox’s S3Gen + **HiFT** path can run on
OpenCL (e.g. Qualcomm Adreno) instead of failing on missing ops:

| What | Purpose |
|------|---------|
| `CONV_TRANSPOSE_1D` | f32 and f16-kernel + f32 input kernels, dispatch + `supports_op` |
| `GGML_OP_SIN` | `sin.cl` (element-wise), dispatch + `supports_op` |
| `ggml-opencl.h` | Document that `ggml_backend_opencl_init` may return NULL when no device |
| Build | Register new `.cl` sources in `CMakeLists.txt` for embed |

Regenerate from a throwaway `ggml` worktree at `GGML_COMMIT` after editing
upstream:

```bash
# From cherry-picked commits or a branch:
(cd ggml && git diff 58c38058..your-branch) > patches/ggml-opencl-chatterbox-ops.patch
# Sanity check on a clean tree:
git -C ggml reset --hard 58c38058 && git -C ggml clean -fdq
git -C ggml apply ../patches/ggml-metal-chatterbox-ops.patch
git -C ggml apply --check ../patches/ggml-opencl-chatterbox-ops.patch
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

* ~~No opportunistic flush during long-running processes.~~ Addressed
  in [`ggml-vulkan-eager-cache-save.patch`](#ggml-vulkan-eager-cache-savepatch)
  (QVAC-17872 round-2).
* We key the filename on `vendorID / deviceID / driverVersion`.  Two
  GPUs of the same model with different driver versions would reuse
  the same filename only when the driver matches, which is the
  intended sharing.
* No attempt to cap the on-disk size.  NVIDIA blobs land at ~1 MB for
  Turbo, ~2 MB for Turbo + MTL shared pipelines.  If this ever becomes
  a concern on mobile, wrap the write with a size check.

## `ggml-vulkan-eager-cache-save.patch`

Base commit: same `58c3805` (stacks on top of `ggml-vulkan-pipeline-cache.patch`).

Closes the crash-safety gap left open by patch 2: it persists the
on-disk pipeline cache **at the end of every `ggml_vk_load_shaders`
invocation that grew the cache**, instead of only at backend-free time
in `ggml_vk_cleanup`.

### Why

`ggml_vk_load_shaders` is the single sync point for both eager
(initial) and lazy (`ggml_pipeline_request_descriptor_sets`) compile
batches.  In chatterbox today **every** pipeline is lazy — the eager
init pass marks nothing `needed`, so `compiles.empty()` returns true on
that call; the actual SPIR-V → driver compile wave happens during the
first graph compute.  Patch 2 catches all of it on the
`ggml_vk_cleanup` save, but **only if the process actually reaches
cleanup**.  A SIGKILL / OOM / crash in the middle of the first
inference throws away seconds of work, and the next run pays the same
cost again.

### What it does

1. Adds `size_t pipeline_cache_last_size` to `vk_device_struct`,
   initialised to the seed-blob size loaded at init.
2. After the `for (auto &c : compiles) c.wait();` loop in
   `ggml_vk_load_shaders`, queries `getPipelineCacheData` and writes it
   to `<path>.tmp` + `rename` **iff `blob.size() > pipeline_cache_last_size`**.
3. Same size-tracked guard is added to `ggml_vk_save_pipeline_cache`
   (the cleanup-time path), so cooperating writers don't fight each
   other and a no-op cleanup costs nothing.

### Why size-only (no hash)

We measured that a naive "flush whenever `compiles.empty() == false`"
path is unsafe on warm runs: `compiles` is non-empty on every lazy
`createComputePipeline` call (even a pure cache hit goes through the
async dispatch path), so the unconditional flush re-writes the same
1 MB blob ~60 times per inference and adds **+90 ms WALL** of disk
churn we measured.  The size compare is the cheapest robust way to
detect "did this call actually grow the on-disk-equivalent blob".  A
SHA256 over the blob would be more robust but costs ~2 ms per check
on the 1 MB blob — disk-write-equivalent overhead — for no extra
correctness on the failure modes we care about (driver bumps invalidate
the seed via Vulkan's own header validation, regardless of size).

### Measured impact (RTX 5090 + NVIDIA driver 590.48)

| Scenario              | Before round-2 | After round-2 | Delta   |
|-----------------------|---------------:|--------------:|--------:|
| Cold first process    |    3.32 s WALL |   3.34 s WALL |  +20 ms (noise) |
| Warm (cache hit)      |    0.89 s WALL |   0.90 s WALL |  +10 ms (noise) |
| Cold + crash recovery |   *3.32 s pay* |  *0.89 s pay* | -2.4 s ≈ 73 % |

The crash-recovery row is qualitative (we kill the process at a known
point during the cold compile wave): without round-2, restart pays the
full cold cost again; with round-2, the partial cache from the crashed
run survives.

### Caveats

* Slightly increased steady-state cold-process disk I/O: each lazy
  compile batch that grows the cache now does an atomic 1 MB write.
  In practice this is ~10-30 writes per cold first-process run,
  totalling a few MB.  Negligible on any storage faster than spinning
  rust.
* `pipeline_cache_last_size` resets per process — if two processes
  race-update the same cache file, last-writer-wins (atomic via
  `rename`).  This is the same property as patch 2 and is the desired
  behaviour for shared per-machine caches.
