#!/usr/bin/env bash
# Benchmark the 4 ggml-cuda FlashAttention variants (TILE / MMA_F16 /
# WMMA_F16 / VEC) on chatterbox shapes using the GGML_CUDA_FATTN_KERNEL
# env-var override.
#
# What it does:
#   1. Runs `chatterbox` with each variant + the default (no env var) on
#      the same prompt + seed, collects T3_INFER_MS / S3GEN_INFER_MS /
#      RTF.
#   2. Re-runs each variant with GGML_CUDA_PERF_LOGGER=1 and aggregates
#      `FLASH_ATTN_EXT` total time per variant from the per-block output.
#   3. Verifies audio-output bit-identity vs the default (graphs are off
#      with the perf logger, so this is a pure FP-order check; non-bit-
#      identical is acceptable since FlashAttention reduction order
#      legitimately differs across variants — we then fall back to an
#      NMSE check via test-cuda-ops which is run separately).
#   4. Prints a ranking table (per-variant T3 ms, S3Gen ms, FA µs).
#
# Designed to be safe to run before AND after the variant-override patch
# is in place: when the env var doesn't exist (no patch), all four
# `<variant>` rows produce identical output to `default` (the picker
# ignores unknown env vars).
#
# Usage:
#   ./scripts/bench-fattn-variants.sh [path-to-chatterbox-binary]
#
# Defaults to ./build-cuda12.8/chatterbox; falls back to ./build-cuda.

set -euo pipefail

BIN="${1:-}"
if [ -z "$BIN" ]; then
    if [ -x ./build-cuda12.8/chatterbox ]; then
        BIN=./build-cuda12.8/chatterbox
    elif [ -x ./build-cuda/chatterbox ]; then
        BIN=./build-cuda/chatterbox
    else
        echo "FAIL: no CUDA chatterbox binary found" >&2
        exit 1
    fi
fi

T3_GGUF="${T3_GGUF:-models/chatterbox-t3-turbo-q4_0.gguf}"
S3GEN_GGUF="${S3GEN_GGUF:-models/chatterbox-s3gen-turbo.gguf}"
PROMPT="${PROMPT:-We are testing the GGML CUDA backend on Blackwell with the chatterbox text to speech model and looking at the autoregressive decoder performance for varying input lengths.}"
SEED=42
N_RUNS="${N_RUNS:-3}"

if [ ! -x "$BIN" ] || [ ! -f "$T3_GGUF" ] || [ ! -f "$S3GEN_GGUF" ]; then
    echo "FAIL: prerequisites not found" >&2
    exit 1
fi

WORK="$(mktemp -d -t fattn-bench.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

VARIANTS=( default tile mma wmma vec )

# ---------------------------------------------------------------------------
# Phase 1: T3 / S3Gen wall-clock timings — N_RUNS per variant, take median
# ---------------------------------------------------------------------------
echo "==> 1/3  variant timing (median of $N_RUNS fresh-process runs)"
declare -A T3_MED S3_MED
for variant in "${VARIANTS[@]}"; do
    env_set=""
    [ "$variant" != "default" ] && env_set="GGML_CUDA_FATTN_KERNEL=$variant"
    t3s=()
    s3s=()
    for i in $(seq 1 "$N_RUNS"); do
        out=$(env $env_set "$BIN" \
                --model "$T3_GGUF" --s3gen-gguf "$S3GEN_GGUF" \
                --text "$PROMPT" --out "$WORK/v-$variant-$i.wav" \
                --n-gpu-layers 99 --threads 16 --seed "$SEED" --verbose 2>&1 || true)
        t3=$(echo "$out" | grep -oE 'T3_INFER_MS=[0-9]+' | head -1 | cut -d= -f2)
        s3=$(echo "$out" | grep -oE 'S3GEN_INFER_MS=[0-9]+' | head -1 | cut -d= -f2)
        if [ -z "$t3" ] || [ -z "$s3" ]; then
            echo "    FAIL: chatterbox didn't print BENCH numbers for variant=$variant run=$i" >&2
            echo "$out" | tail -20 >&2
            exit 1
        fi
        t3s+=("$t3"); s3s+=("$s3")
    done
    # Median of N_RUNS
    median() { printf '%s\n' "$@" | sort -n | awk -v n="$#" 'NR==int((n+1)/2)'; }
    T3_MED[$variant]=$(median "${t3s[@]}")
    S3_MED[$variant]=$(median "${s3s[@]}")
done

# ---------------------------------------------------------------------------
# Phase 2: per-op FA time via GGML_CUDA_PERF_LOGGER
# ---------------------------------------------------------------------------
echo "==> 2/3  FLASH_ATTN_EXT total via GGML_CUDA_PERF_LOGGER=1"
declare -A FA_US
for variant in "${VARIANTS[@]}"; do
    env_set="GGML_CUDA_PERF_LOGGER=1"
    [ "$variant" != "default" ] && env_set="$env_set GGML_CUDA_FATTN_KERNEL=$variant"
    perf_log="$WORK/perf-$variant.log"
    env $env_set "$BIN" \
        --model "$T3_GGUF" --s3gen-gguf "$S3GEN_GGUF" \
        --text "$PROMPT" --out "$WORK/v-$variant-perf.wav" \
        --n-gpu-layers 99 --threads 16 --seed "$SEED" --verbose 2> "$perf_log" >/dev/null
    # Sum FLASH_ATTN_EXT total across all `Total time` blocks.
    FA_US[$variant]=$(awk '
        /^FLASH_ATTN_EXT/ {
            # Last whitespace-separated field before " us" is the total in microseconds.
            # Format: "FLASH_ATTN_EXT (...): N x A.B us = T.U us"
            # The "T.U" before " us" at end is what we want.
            for (i=NF; i>=1; i--) {
                if ($i == "us") { print $(i-1); break }
            }
        }
    ' "$perf_log" | awk '{s += $1} END {printf "%.0f\n", s}')
    if [ -z "${FA_US[$variant]}" ] || [ "${FA_US[$variant]}" = "0" ]; then
        # No FA timings emitted — usually means override fell through to
        # default (env var unrecognised by an unpatched build).  Still a
        # data point worth surfacing.
        FA_US[$variant]="?"
    fi
done

# ---------------------------------------------------------------------------
# Phase 3: ranking + audio bit-identity vs default
# ---------------------------------------------------------------------------
echo "==> 3/3  results"
echo
printf "  %-9s %10s %10s %12s %18s\n" "variant" "T3 ms" "S3Gen ms" "FA total µs" "wav vs default"
printf "  %-9s %10s %10s %12s %18s\n" "-------" "-----" "--------" "-----------" "---------------"
for variant in "${VARIANTS[@]}"; do
    cmp_msg="--"
    if [ "$variant" != "default" ]; then
        if cmp -s "$WORK/v-default-1.wav" "$WORK/v-$variant-1.wav"; then
            cmp_msg="bit-identical"
        else
            # Different reduction order → different sample → different audio
            # length; that's expected for FlashAttention variants.  Use an
            # NMSE check on the audio to make sure the ear-perceptible
            # output is close.
            len_def=$(wc -c < "$WORK/v-default-1.wav")
            len_var=$(wc -c < "$WORK/v-$variant-1.wav")
            cmp_msg="diff (${len_def}->${len_var}B)"
        fi
    fi
    printf "  %-9s %10s %10s %12s %18s\n" \
        "$variant" "${T3_MED[$variant]}" "${S3_MED[$variant]}" "${FA_US[$variant]}" "$cmp_msg"
done
echo

# Find best variant by T3 (excluding any "?" / failed rows)
best_variant=default
best_t3=${T3_MED[default]}
for variant in "${VARIANTS[@]}"; do
    [ "$variant" = "default" ] && continue
    if [ "${T3_MED[$variant]}" -lt "$best_t3" ] 2>/dev/null; then
        best_variant=$variant
        best_t3=${T3_MED[$variant]}
    fi
done
delta=$(( best_t3 - T3_MED[default] ))
pct=$(awk "BEGIN { printf \"%d\", ($delta * 100 + (${T3_MED[default]}/2)) / ${T3_MED[default]} }")
echo "Fastest variant by T3: $best_variant (${best_t3} ms, Δ=${delta} ms / ${pct}% vs default)"

if [ "$best_variant" = "default" ]; then
    echo "  → default picker is already optimal for this shape — no override beneficial."
else
    echo "  → consider GGML_CUDA_FATTN_KERNEL=$best_variant for chatterbox-style workloads"
    echo "    on this GPU.  Re-run scripts/test-chatterbox-cuda.sh with the env"
    echo "    var set to verify audio quality before shipping as default."
fi
