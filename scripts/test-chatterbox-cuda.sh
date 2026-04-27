#!/usr/bin/env bash
# End-to-end CUDA smoke test for chatterbox.cpp.
#
# Validates the high-level invariants that the patches in
# patches/ggml-cuda-chatterbox-ops.patch must preserve:
#
#   1. GGML_CUDA_FORCE_GRAPHS=1 produces *bit-identical* audio vs the
#      default CUDA path (graph orchestration changes only, no math).
#   2. Chatterbox runs without crashing across a few seeds × prompt
#      lengths (regression guard for the warp-cooperative
#      conv_transpose_1d kernel — large grid sizes etc.).
#   3. FORCE_GRAPHS does not regress T3 perf (within ±5 %).
#
# Kernel-level CPU-vs-GPU correctness for the patched conv_transpose_1d
# is covered by build-cuda/test-cuda-ops; this script tests the wired-up
# pipeline behaviour.  End-to-end CPU-vs-CUDA audio equivalence is *not*
# checked because chatterbox's autoregressive sampler diverges with
# backend FP precision (different token counts → different audio
# lengths), which is expected for any generative model.
#
# Usage:
#   ./scripts/test-chatterbox-cuda.sh [path-to-chatterbox-binary]
#
# Defaults to ./build-cuda/chatterbox.  Exits non-zero on any failure.

set -euo pipefail

BIN="${1:-./build-cuda/chatterbox}"
T3_GGUF="${T3_GGUF:-models/chatterbox-t3-turbo-q4_0.gguf}"
S3GEN_GGUF="${S3GEN_GGUF:-models/chatterbox-s3gen-turbo.gguf}"
PROMPT='Hello from ggml.'
SEED=42

if [ ! -x "$BIN" ]; then
    echo "FAIL: chatterbox binary not found at $BIN" >&2
    echo "  build with: cmake --build build-cuda -j --target chatterbox" >&2
    exit 1
fi
if [ ! -f "$T3_GGUF" ] || [ ! -f "$S3GEN_GGUF" ]; then
    echo "FAIL: model GGUFs not found" >&2
    echo "  expected: $T3_GGUF" >&2
    echo "            $S3GEN_GGUF" >&2
    echo "  see scripts/convert-t3-turbo-to-gguf.py / convert-s3gen-to-gguf.py" >&2
    exit 1
fi

WAV_DIR="$(mktemp -d -t chbx-cuda-test.XXXXXX)"
trap 'rm -rf "$WAV_DIR"' EXIT

run_chatterbox() {
    local label="$1"; shift
    local out="$WAV_DIR/$label.wav"
    env "$@" "$BIN" \
        --model       "$T3_GGUF" \
        --s3gen-gguf  "$S3GEN_GGUF" \
        --text        "$PROMPT" \
        --out         "$out" \
        --n-gpu-layers 99 --threads 16 --seed "$SEED" --verbose 2>&1
    echo "$out"
}

extract_t3_ms() {
    grep -oE 'T3_INFER_MS=[0-9]+' "$1" | head -1 | cut -d= -f2
}

# ---------------------------------------------------------------------------
# 1. FORCE_GRAPHS bit-identity (default vs FORCE_GRAPHS, same seed/prompt)
# ---------------------------------------------------------------------------
echo "==> 1/4  FORCE_GRAPHS bit-identity check"
LOG_DEFAULT="$WAV_DIR/default.log"
run_chatterbox default > "$LOG_DEFAULT"
WAV_DEFAULT="$WAV_DIR/default.wav"
T3_DEFAULT="$(extract_t3_ms "$LOG_DEFAULT")"

LOG_FORCE="$WAV_DIR/force.log"
run_chatterbox force GGML_CUDA_FORCE_GRAPHS=1 > "$LOG_FORCE"
WAV_FORCE="$WAV_DIR/force.wav"
T3_FORCE="$(extract_t3_ms "$LOG_FORCE")"

echo "    default T3=${T3_DEFAULT} ms  force T3=${T3_FORCE} ms  ($(wc -c < "$WAV_DEFAULT") bytes)"
if cmp -s "$WAV_DEFAULT" "$WAV_FORCE"; then
    echo "    PASS: FORCE_GRAPHS audio bit-identical to default"
else
    echo "    FAIL: FORCE_GRAPHS audio differs from default" >&2
    md5sum "$WAV_DEFAULT" "$WAV_FORCE" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Stress: 3 seeds × 3 prompt lengths in both modes (regression guard
#    for the warp-cooperative conv_transpose_1d kernel + crash check)
# ---------------------------------------------------------------------------
echo "==> 2/4  stress runs (3 seeds × 3 prompts × 2 modes = 18 runs)"
SHORT='Hi.'
MID='Hello from ggml.'
LONG='We are testing the GGML CUDA backend on Blackwell with the chatterbox text to speech model.'
run_quiet() {
    local seed="$1"; local prompt="$2"; local mode="$3"
    local out="$WAV_DIR/stress-${seed}-${mode}-$(echo "$prompt" | head -c 6 | tr -dc 'a-zA-Z0-9').wav"
    local env_set=()
    [ "$mode" = "force" ] && env_set+=(GGML_CUDA_FORCE_GRAPHS=1)
    if ! env "${env_set[@]+"${env_set[@]}"}" "$BIN" \
            --model "$T3_GGUF" --s3gen-gguf "$S3GEN_GGUF" \
            --text "$prompt" --out "$out" \
            --n-gpu-layers 99 --threads 16 --seed "$seed" >/dev/null 2>&1; then
        echo "    FAIL: chatterbox crashed (seed=$seed mode=$mode prompt='$prompt')" >&2
        return 1
    fi
    if [ ! -s "$out" ]; then
        echo "    FAIL: empty wav (seed=$seed mode=$mode prompt='$prompt')" >&2
        return 1
    fi
}
for seed in 1 42 99; do
    for prompt in "$SHORT" "$MID" "$LONG"; do
        for mode in default force; do
            run_quiet "$seed" "$prompt" "$mode"
        done
    done
done
echo "    PASS: 18/18 stress runs completed without crashes / empty output"

# ---------------------------------------------------------------------------
# 3. Perf sanity: FORCE_GRAPHS should not regress T3 vs default (>5 %)
# ---------------------------------------------------------------------------
echo "==> 3/4  perf sanity"
if [ "${T3_DEFAULT:-0}" -gt 0 ] && [ "${T3_FORCE:-0}" -gt 0 ]; then
    DELTA=$(( T3_FORCE - T3_DEFAULT ))
    PCT=$(awk "BEGIN { printf \"%d\", ($DELTA * 100 + ($T3_DEFAULT/2)) / $T3_DEFAULT }")
    echo "    T3 default=${T3_DEFAULT}ms  force=${T3_FORCE}ms  Δ=${DELTA}ms (${PCT}%)"
    if [ "$PCT" -gt 5 ]; then
        echo "    FAIL: FORCE_GRAPHS regressed T3 by >5 %" >&2
        exit 1
    fi
    echo "    PASS: FORCE_GRAPHS T3 within tolerance (or faster)"
else
    echo "    SKIP: could not parse T3 timings"
fi

# ---------------------------------------------------------------------------
# 4. Env-var combination matrix.  All combinations must produce a
#    valid wav file with no crash; FORCE_GRAPHS-default-disable
#    semantics interact with DISABLE_GRAPHS / DISABLE_FUSION /
#    PERF_LOGGER, so cover the cross-product.
# ---------------------------------------------------------------------------
echo "==> 4/4  env-var combination matrix"
# (slug, env-string) pairs.  Slug is used as the wav filename.
COMBO_SLUGS=( default       force                    nofuse                  force_nofuse                                              disable_then_force                                              perflog                  perflog_force                                  )
COMBO_ENVS=(  ""            "GGML_CUDA_FORCE_GRAPHS=1" "GGML_CUDA_DISABLE_FUSION=1" "GGML_CUDA_FORCE_GRAPHS=1 GGML_CUDA_DISABLE_FUSION=1" "GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_FORCE_GRAPHS=1"            "GGML_CUDA_PERF_LOGGER=1" "GGML_CUDA_PERF_LOGGER=1 GGML_CUDA_FORCE_GRAPHS=1" )
n_total=${#COMBO_SLUGS[@]}
n_passed=0
for i in "${!COMBO_SLUGS[@]}"; do
    slug="${COMBO_SLUGS[$i]}"
    combo="${COMBO_ENVS[$i]}"
    out_wav="$WAV_DIR/combo-${slug}.wav"
    if env $combo "$BIN" \
            --model "$T3_GGUF" --s3gen-gguf "$S3GEN_GGUF" \
            --text "$PROMPT" --out "$out_wav" \
            --n-gpu-layers 99 --threads 16 --seed "$SEED" >/dev/null 2>&1; then
        if [ -s "$out_wav" ]; then
            n_passed=$((n_passed + 1))
        else
            echo "    FAIL: empty wav for combo: '$slug' ('$combo')" >&2
            exit 1
        fi
    else
        echo "    FAIL: chatterbox crashed for combo: '$slug' ('$combo')" >&2
        exit 1
    fi
done
echo "    PASS: ${n_passed}/${n_total} env-var combinations completed without crash / empty output"

# Bit-identity invariants on top of the matrix:
# (a) FORCE_GRAPHS doesn't change math vs default → wav identical
# (b) FORCE_GRAPHS doesn't change math when fusion is off either
# (c) DISABLE_GRAPHS overrides FORCE_GRAPHS (the latter is contingent
#     on graphs being enabled in the first place) → identical to a
#     run that just doesn't try to use graphs
check_identical() {
    local label="$1" a="$2" b="$3"
    if cmp -s "$a" "$b"; then
        echo "    PASS: $label — $(basename $a) == $(basename $b)"
    else
        echo "    FAIL: $label — wavs differ" >&2
        md5sum "$a" "$b" >&2
        exit 1
    fi
}
check_identical "(a) FORCE_GRAPHS bit-identical to default"      \
    "$WAV_DIR/combo-default.wav" "$WAV_DIR/combo-force.wav"
check_identical "(b) FORCE_GRAPHS bit-identical with DISABLE_FUSION" \
    "$WAV_DIR/combo-nofuse.wav" "$WAV_DIR/combo-force_nofuse.wav"
check_identical "(c) DISABLE_GRAPHS overrides FORCE_GRAPHS"       \
    "$WAV_DIR/combo-default.wav" "$WAV_DIR/combo-disable_then_force.wav"

echo
echo "All chatterbox.cpp CUDA smoke tests PASSED."
