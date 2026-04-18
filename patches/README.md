# ggml patches for Chatterbox

`ggml` is vendored as a fresh upstream clone (see the top-level README), so
any fixes we need in it live here as standalone patches and are applied
after the clone.

## Apply

```bash
# From the repo root, after cloning upstream ggml into ./ggml
cd ggml
git apply ../patches/ggml-metal-chatterbox-ops.patch
cd ..
# then rebuild as usual
```

## `ggml-metal-chatterbox-ops.patch`

Base commit: `58c3805` (`sync : llama.cpp`, 2026-04-09).

Fixes three gaps in ggml-metal that make Chatterbox unusable or very slow
on Metal:

| Symptom                                       | Root cause in ggml-metal                          | What this patch does                                           |
|-----------------------------------------------|---------------------------------------------------|----------------------------------------------------------------|
| T3 crashes: `unsupported op 'DIAG_MASK_INF'`  | No op entry / no kernel                            | Adds `kernel_diag_mask_inf_f32`, dispatcher, `supports_op` case|
| S3Gen crashes: `unsupported op 'PAD'` when any front-pad (`lp0..lp3`) is non-zero | Kernel only supports tail padding; `supports_op` rejects non-zero front pads | Extends `kernel_pad_f32` + `ggml_metal_kargs_pad` to honour `lp0..lp3` and drops the rejection  |
| HiFT decode is ~100× slower than CPU          | `kernel_conv_transpose_1d` is scalar: 1 thread per output pixel iterating over *all* `IC * IL` inputs, with most of the work inside a conditional | Tighten the input-position range to the few that contribute (`i_min..i_max`) and parallelise `IC` across a 32-thread simdgroup with `simd_sum` reduction |

Measured on M3 Ultra, `hift_decode` at HiFT-realistic shapes:
- Before: ~15 000 ms
- After:    ~350 ms (≈ 40× speedup; end-to-end `gen_RTF` goes from unusable → 0.19 on F16)

Correctness is validated against the ggml CPU backend by
`build/test-metal-ops` (added in the parent repo).

## Dropping the patch

If upstream ggml merges equivalent fixes, delete the patch file and
remove the `git apply` step from the build instructions. The C++ side
of Chatterbox already uses ops supported by every backend, so nothing
else needs to change.
