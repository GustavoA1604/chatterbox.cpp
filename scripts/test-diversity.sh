#!/usr/bin/env bash
# Diversity / edge-case regression for the chatterbox CUDA build.
#
# The 50-run soak (test-stability.sh) only validates one prompt + one seed
# repeated 50 times. This script complements it by sweeping
# real-world variation:
#
#   1. Different SEEDS (changes the autoregressive sample path → exercises
#      KV-cache + flash-attn + MUL_MAT_VEC fusion under different token streams).
#   2. Multilingual / non-ASCII PROMPTS (UTF-8 BPE path; Latin / German /
#      French / Spanish — Chatterbox-multilingual is the model under test).
#   3. Edge-case DURATIONS:
#        - very short prompt   (single word, ~2 tokens, fastest path)
#        - typical prompt      (sentence, ~30 tokens)
#        - long prompt         (~50 words, exercises growing KV cache)
#        - empty-ish whitespace (graceful handling, no crash)
#
# All runs use the same env (default config — fusion ON, graphs warmup).
# We assert the binary exits 0 and emits a non-empty wav for every
# (seed, prompt) pair. No bit-identity check across different inputs
# (they SHOULD differ); we just want to catch crashes / silent corruption /
# zero-length output that would only show up on inputs we haven't tried.
#
# Usage:
#   ./scripts/test-diversity.sh [path-to-chatterbox-binary]

set -euo pipefail

BIN="${1:-./build-cuda12.8/chatterbox}"
T3_GGUF="${T3_GGUF:-models/chatterbox-t3-turbo-q4_0.gguf}"
S3GEN_GGUF="${S3GEN_GGUF:-models/chatterbox-s3gen-turbo.gguf}"

if [ ! -x "$BIN" ] || [ ! -f "$T3_GGUF" ] || [ ! -f "$S3GEN_GGUF" ]; then
    echo "FAIL: prerequisites not found (binary or models)" >&2
    exit 1
fi

WORK="$(mktemp -d -t chbx-div.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

# 1) Seeds (5 distinct values incl. 0 default and a large prime).
SEEDS=(0 1 7 42 1000003)

# 2) Multilingual prompts (UTF-8). Chatterbox-multilingual handles these.
declare -a PROMPTS=(
    "Hello from ggml."
    "Bonjour, comment allez-vous aujourd'hui."
    "Hola, esto es una prueba del decodificador."
    "Guten Tag, wir testen das System."
    "Ciao, questo è un test di sintesi vocale."
)

# 3) Edge-case durations (kept ASCII to isolate the length axis from i18n).
declare -a EDGE_PROMPTS=(
    "Hi."
    "We are testing the GGML CUDA backend with the chatterbox text to speech model and looking at the autoregressive decoder performance for varying input lengths and various edge cases that might occur in production environments."
    "  word   "
)

run_one() {
    local label="$1"; shift
    local seed="$1"; shift
    local prompt="$1"; shift
    local out_wav="$WORK/${label}.wav"
    local log_file="$WORK/${label}.log"
    if ! "$BIN" \
            --model "$T3_GGUF" --s3gen-gguf "$S3GEN_GGUF" \
            --text "$prompt" --out "$out_wav" \
            --n-gpu-layers 99 --threads 16 --seed "$seed" --verbose \
        > "$log_file" 2>&1; then
        echo "    FAIL: $label crashed" >&2
        tail -15 "$log_file" >&2
        return 1
    fi
    if [ ! -s "$out_wav" ]; then
        echo "    FAIL: $label produced empty wav" >&2
        return 1
    fi
    local sz
    sz=$(stat -c '%s' "$out_wav")
    if [ "$sz" -lt 1024 ]; then
        echo "    FAIL: $label produced suspiciously small wav (${sz} bytes)" >&2
        return 1
    fi
    local t3
    t3=$(grep -oE 'T3_INFER_MS=[0-9]+' "$log_file" | head -1 | cut -d= -f2)
    echo "    [$label]  seed=${seed}  T3=${t3}ms  wav=${sz}B"
    return 0
}

# ---------------------------------------------------------------------------
# 1. Seed sweep (fixed prompt, varying seeds)
# ---------------------------------------------------------------------------
echo "==> 1/3  seed sweep (5 seeds, fixed prompt)"
SEED_MD5S=""
for seed in "${SEEDS[@]}"; do
    label="seed-$seed"
    run_one "$label" "$seed" "${PROMPTS[0]}" || exit 1
    SEED_MD5S="$SEED_MD5S $(md5sum "$WORK/${label}.wav" | cut -d' ' -f1)"
done
unique_seed=$(echo $SEED_MD5S | tr ' ' '\n' | sort -u | wc -l)
if [ "$unique_seed" -lt 2 ]; then
    echo "    FAIL: different seeds produced identical output (got $unique_seed unique md5s, expected >=2)" >&2
    exit 1
fi
echo "    PASS: seed sweep produced ${unique_seed}/${#SEEDS[@]} distinct outputs (sampler responding to seed)"

# ---------------------------------------------------------------------------
# 2. Multilingual prompt sweep (5 languages, fixed seed)
# ---------------------------------------------------------------------------
echo "==> 2/3  multilingual prompts (5 languages, seed=42)"
LANG_MD5S=""
for i in "${!PROMPTS[@]}"; do
    run_one "lang-$i" 42 "${PROMPTS[$i]}" || exit 1
    LANG_MD5S="$LANG_MD5S $(md5sum "$WORK/lang-$i.wav" | cut -d' ' -f1)"
done
unique_lang=$(echo $LANG_MD5S | tr ' ' '\n' | sort -u | wc -l)
if [ "$unique_lang" -ne ${#PROMPTS[@]} ]; then
    echo "    FAIL: distinct prompts collided to ${unique_lang} unique outputs (expected ${#PROMPTS[@]})" >&2
    exit 1
fi
echo "    PASS: each prompt produced a distinct wav"

# ---------------------------------------------------------------------------
# 3. Edge-case duration sweep (very short / very long / whitespace)
# ---------------------------------------------------------------------------
echo "==> 3/3  edge-case durations"
for i in "${!EDGE_PROMPTS[@]}"; do
    run_one "edge-$i" 42 "${EDGE_PROMPTS[$i]}" || exit 1
done
echo "    PASS: edge-case durations produced non-empty wavs"

echo
echo "All chatterbox.cpp CUDA diversity tests PASSED."
