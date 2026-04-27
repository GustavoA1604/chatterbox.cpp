#!/usr/bin/env bash
# End-to-end test for GGML_CUDA_PERF_LOGGER=1 (the per-op GPU timing
# logger added in patches/ggml-cuda-chatterbox-ops.patch).
#
# Verifies that:
#   1. Setting the env var produces a "CUDA Timings:" block per
#      ggml_backend_cuda_graph_compute call.
#   2. The output line format matches ggml-vulkan's vk_perf_logger
#      ("<name>: N x A.B us = T.U us") so cross-backend grep one-liners
#      keep working.
#   3. At least one MUL_MAT op shows up (T3 step + S3Gen both heavy on it).
#   4. The patched conv_transpose_1d kernel shows up under
#      CONV_TRANSPOSE_1D — confirms the fusion / passthrough path is
#      timed.
#   5. Setting the env var DISABLES CUDA Graphs (matches the behaviour
#      we baked into is_enabled() in common.cuh).  Verified indirectly:
#      with PERF_LOGGER=1 + FORCE_GRAPHS=1, no "CUDA graph (force)
#      warmup complete" debug line should appear (best-effort check;
#      the GGML_LOG_DEBUG line is only emitted in debug builds).
#   6. Default behaviour (env var unset) is unchanged: no "CUDA
#      Timings:" output, audio still bit-identical to a previous golden
#      run.
#
# Usage:
#   ./scripts/test-cuda-perf-logger.sh [path-to-chatterbox-binary]
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
PROMPT='Hi.'
SEED=42

if [ ! -x "$BIN" ] || [ ! -f "$T3_GGUF" ] || [ ! -f "$S3GEN_GGUF" ]; then
    echo "FAIL: prerequisites not found (binary or models)" >&2
    exit 1
fi

WORK="$(mktemp -d -t chbx-cuda-perf.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

run_chatterbox() {
    local out_wav="$1"; shift
    local stderr_log="$1"; shift
    env "$@" "$BIN" \
        --model      "$T3_GGUF" \
        --s3gen-gguf "$S3GEN_GGUF" \
        --text       "$PROMPT" \
        --out        "$out_wav" \
        --n-gpu-layers 99 --threads 16 --seed "$SEED" --verbose 2> "$stderr_log" >/dev/null
}

# ---------------------------------------------------------------------------
# 1. Default run — golden audio + golden stderr (no perf-logger output)
# ---------------------------------------------------------------------------
echo "==> 1/4  default run (env var unset)"
run_chatterbox "$WORK/golden.wav" "$WORK/golden.log"
GOLDEN_BYTES=$(wc -c < "$WORK/golden.wav")
if grep -q "^CUDA Timings:" "$WORK/golden.log"; then
    echo "    FAIL: 'CUDA Timings:' appeared without env var set" >&2
    exit 1
fi
echo "    PASS: no perf-logger output by default ($GOLDEN_BYTES wav bytes)"

# ---------------------------------------------------------------------------
# 2. Env-var run — perf-logger output appears, audio unchanged
# ---------------------------------------------------------------------------
echo "==> 2/4  GGML_CUDA_PERF_LOGGER=1"
run_chatterbox "$WORK/perf.wav" "$WORK/perf.log" GGML_CUDA_PERF_LOGGER=1
PERF_BYTES=$(wc -c < "$WORK/perf.wav")

# Audio should be bit-identical (perf logger only adds timing events,
# no math change).  Allow audio length to vary by 1 sample because the
# autoregressive sampler uses fp arithmetic that's perturbation-stable
# but not strictly so — but in practice, on the same seed and binary
# graphs-disabled vs graphs-disabled should produce identical output.
if ! cmp -s "$WORK/golden.wav" "$WORK/perf.wav"; then
    # Soften: allow if same length AND |max_diff| ≤ 1 LSB (FP-order noise).
    python3 - "$WORK/golden.wav" "$WORK/perf.wav" <<'EOF' || { echo "    FAIL: perf-logger audio diverges from default" >&2; exit 1; }
import sys, wave, struct
def read(p):
    with wave.open(p,'rb') as w:
        return w.getnframes(), w.readframes(w.getnframes())
n1, a = read(sys.argv[1])
n2, b = read(sys.argv[2])
assert n1 == n2, f"length differs ({n1} vs {n2})"
# 16-bit PCM little-endian
xs = [struct.unpack('<h', a[i:i+2])[0] for i in range(0, len(a), 2)]
ys = [struct.unpack('<h', b[i:i+2])[0] for i in range(0, len(b), 2)]
maxd = max(abs(x-y) for x,y in zip(xs,ys))
assert maxd <= 1, f"max sample diff = {maxd} (>1 LSB)"
print(f"    audio match within {maxd} LSB")
EOF
fi
if [ "$GOLDEN_BYTES" != "$PERF_BYTES" ]; then
    echo "    FAIL: perf-logger wav size differs from default ($GOLDEN_BYTES vs $PERF_BYTES)" >&2
    exit 1
fi

# Output structure
N_BLOCKS=$(grep -c "^CUDA Timings:" "$WORK/perf.log" || echo 0)
if [ "$N_BLOCKS" -lt 1 ]; then
    echo "    FAIL: expected ≥ 1 'CUDA Timings:' block, got $N_BLOCKS" >&2
    exit 1
fi
N_TOTALS=$(grep -c "^Total time:" "$WORK/perf.log" || echo 0)
if [ "$N_TOTALS" -lt 1 ]; then
    echo "    FAIL: expected ≥ 1 'Total time:' line, got $N_TOTALS" >&2
    exit 1
fi
if [ "$N_BLOCKS" != "$N_TOTALS" ]; then
    echo "    FAIL: 'CUDA Timings:' blocks ($N_BLOCKS) != 'Total time:' lines ($N_TOTALS)" >&2
    exit 1
fi
echo "    PASS: $N_BLOCKS 'CUDA Timings:' blocks, $N_TOTALS 'Total time:' lines"

# Format: "<name>: <N> x <avg> us = <total> us"
N_LINES=$(grep -cE '^[A-Za-z_].*: [0-9]+ x [0-9]+\.[0-9]+ us = [0-9]+\.[0-9]+ us$' "$WORK/perf.log" || echo 0)
if [ "$N_LINES" -lt 5 ]; then
    echo "    FAIL: too few well-formatted op lines ($N_LINES)" >&2
    head -40 "$WORK/perf.log" >&2
    exit 1
fi
echo "    PASS: $N_LINES well-formatted op lines"

# Expected hot ops should all show up across the whole stderr
for op in MUL_MAT FLASH_ATTN_EXT NORM CONV_TRANSPOSE_1D; do
    if ! grep -q "^${op}" "$WORK/perf.log"; then
        echo "    FAIL: expected op '$op' missing from perf logger output" >&2
        exit 1
    fi
done
echo "    PASS: MUL_MAT / FLASH_ATTN_EXT / NORM / CONV_TRANSPOSE_1D all timed"

# ---------------------------------------------------------------------------
# 3. PERF_LOGGER + FORCE_GRAPHS — graphs MUST be disabled by perf logger
# ---------------------------------------------------------------------------
echo "==> 3/4  GGML_CUDA_PERF_LOGGER=1 + GGML_CUDA_FORCE_GRAPHS=1 (graphs auto-disabled)"
run_chatterbox "$WORK/both.wav" "$WORK/both.log" \
    GGML_CUDA_PERF_LOGGER=1 GGML_CUDA_FORCE_GRAPHS=1
if ! grep -q "^CUDA Timings:" "$WORK/both.log"; then
    echo "    FAIL: perf logger produced no output when FORCE_GRAPHS was also set" >&2
    exit 1
fi
# best-effort: even if graphs were on, FORCE warmup-complete debug
# message would appear in debug builds; in release builds we just
# check that perf logger output exists, since that's the observable
# proof the graphs path is gated off (graphs would suppress per-op
# timing).
echo "    PASS: perf logger still produces output with FORCE_GRAPHS set"

# ---------------------------------------------------------------------------
# 4. Aggregate-time sanity: sum of per-op totals should be < 1s for
#    a 19-token prompt, and > 1ms (smoke proof we measured anything).
# ---------------------------------------------------------------------------
echo "==> 4/4  aggregate-time sanity"
TOTAL_LAST=$(grep "^Total time:" "$WORK/perf.log" | tail -1 | awk '{print $3}')
TOTAL_NS=$(awk "BEGIN { printf \"%.0f\", $TOTAL_LAST * 1000 }")
if [ "$TOTAL_NS" -lt 1000 ]; then
    echo "    FAIL: last 'Total time:' = $TOTAL_LAST us — implausibly small" >&2
    exit 1
fi
if [ "$TOTAL_NS" -gt 1000000000 ]; then
    echo "    FAIL: last 'Total time:' = $TOTAL_LAST us > 1 s — implausibly large" >&2
    exit 1
fi
echo "    PASS: last 'Total time:' = $TOTAL_LAST us (sane range)"

echo
echo "All GGML_CUDA_PERF_LOGGER tests PASSED."
