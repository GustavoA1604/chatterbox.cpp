#!/usr/bin/env bash
# Stability + memory-leak regression for chatterbox CUDA build.
#
# Production-style soak: runs `chatterbox` N times in sequence with
# the same prompt + seed, asserts:
#
#   1. Every run produces the SAME output (bit-identical wav across all
#      runs — proves the autoregressive sampler is deterministic and no
#      run-to-run state leaks between processes).
#   2. T3 / S3Gen wall times don't drift upwards by > 5 % from runs 5-9
#      to the last 5 runs (catches per-run cache growth, GPU memory
#      fragmentation, pool-exhaustion regressions).
#   3. RSS doesn't grow by more than 10 MB from runs 5-9 to last 5 runs
#      (catches process-level memory leaks; allows a small slack for
#      libc heap stabilisation).
#   4. NVIDIA driver `~/.nv/ComputeCache/` size stays bounded
#      (catches accidental JIT recompiles that shouldn't happen on a
#      stable native-SASS build).
#
# Skips the first 5 runs in the time / RSS regression checks because
# CUDA driver init, library load, and the first cudaMalloc warmup
# legitimately settle over the first few invocations.
#
# Usage:
#   ./scripts/test-stability.sh [path-to-chatterbox-binary] [n-runs]
#
# Defaults: ./build-cuda12.8/chatterbox, 50 runs.

set -euo pipefail

BIN="${1:-./build-cuda12.8/chatterbox}"
N_RUNS="${2:-50}"

T3_GGUF="${T3_GGUF:-models/chatterbox-t3-turbo-q4_0.gguf}"
S3GEN_GGUF="${S3GEN_GGUF:-models/chatterbox-s3gen-turbo.gguf}"
PROMPT="${PROMPT:-Hello from ggml.}"
SEED=42

if [ ! -x "$BIN" ] || [ ! -f "$T3_GGUF" ] || [ ! -f "$S3GEN_GGUF" ]; then
    echo "FAIL: prerequisites not found (binary or models)" >&2
    exit 1
fi

WORK="$(mktemp -d -t chbx-stab.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

NV_CACHE="${HOME}/.nv/ComputeCache"
NV_BEFORE_KB=$(du -sk "$NV_CACHE" 2>/dev/null | cut -f1 || echo 0)

echo "==> running $N_RUNS sequential chatterbox processes"
T3_LIST=""
S3_LIST=""
RSS_LIST=""
MD5_LIST=""
for i in $(seq 1 "$N_RUNS"); do
    out_wav="$WORK/run-$i.wav"
    log_file="$WORK/run-$i.log"
    /usr/bin/time -v -o "$WORK/time-$i.txt" \
        "$BIN" \
            --model "$T3_GGUF" --s3gen-gguf "$S3GEN_GGUF" \
            --text "$PROMPT" --out "$out_wav" \
            --n-gpu-layers 99 --threads 16 --seed "$SEED" --verbose \
        > "$log_file" 2>&1 || {
            echo "FAIL: chatterbox crashed on run $i" >&2
            tail -20 "$log_file" >&2
            exit 1
        }
    t3=$(grep -oE 'T3_INFER_MS=[0-9]+' "$log_file" | head -1 | cut -d= -f2)
    s3=$(grep -oE 'S3GEN_INFER_MS=[0-9]+' "$log_file" | head -1 | cut -d= -f2)
    rss=$(awk '/Maximum resident set size/ {print $NF}' "$WORK/time-$i.txt")
    md5=$(md5sum "$out_wav" | cut -d' ' -f1)
    T3_LIST="$T3_LIST $t3"
    S3_LIST="$S3_LIST $s3"
    RSS_LIST="$RSS_LIST $rss"
    MD5_LIST="$MD5_LIST $md5"
    if [ "$((i % 10))" -eq 0 ]; then
        echo "    [$i/$N_RUNS] T3=${t3}ms S3Gen=${s3}ms RSS=${rss}KB"
    fi
done

NV_AFTER_KB=$(du -sk "$NV_CACHE" 2>/dev/null | cut -f1 || echo 0)
NV_DELTA_KB=$(( NV_AFTER_KB - NV_BEFORE_KB ))

# ---------------------------------------------------------------------------
# 1. Bit-identity across all runs
# ---------------------------------------------------------------------------
echo "==> 1/4  bit-identity across $N_RUNS runs"
unique=$(echo $MD5_LIST | tr ' ' '\n' | sort -u | wc -l)
first_md5=$(echo $MD5_LIST | awk '{print $1}')
if [ "$unique" -ne 1 ]; then
    echo "    FAIL: outputs not deterministic (got $unique distinct md5s across $N_RUNS runs)" >&2
    echo $MD5_LIST | tr ' ' '\n' | sort -u | head >&2
    exit 1
fi
echo "    PASS: all $N_RUNS runs produced identical wav (md5: $first_md5)"

# Helper: median of a slice of a space-separated number list (1-indexed, inclusive)
slice_median() {
    local list="$1" start="$2" end="$3"
    echo "$list" \
        | tr ' ' '\n' \
        | sed '/^$/d' \
        | sed -n "${start},${end}p" \
        | sort -n \
        | awk 'BEGIN { n=0 } { a[++n]=$1 } END { if (n==0) exit 1; print a[int((n+1)/2)] }'
}

# Pick comparison windows scaled to N_RUNS.
# For small N (smoke test: 5), use first half vs second half so the script
# still validates end-to-end. For N>=20 use the original "skip 5-warmup
# vs last 5" which gives a sharper signal on real soak runs.
if [ "$N_RUNS" -ge 20 ]; then
    EARLY_START=5;          EARLY_END=9
    LATE_START=$((N_RUNS-4)); LATE_END="$N_RUNS"
else
    HALF=$(( (N_RUNS + 1) / 2 ))
    EARLY_START=1;          EARLY_END="$HALF"
    LATE_START=$((HALF+1)); LATE_END="$N_RUNS"
fi

# ---------------------------------------------------------------------------
# 2. Perf drift (early window → late window)
# ---------------------------------------------------------------------------
echo "==> 2/4  perf drift runs ${EARLY_START}-${EARLY_END} → ${LATE_START}-${LATE_END}"
T3_EARLY=$(slice_median "$T3_LIST" "$EARLY_START" "$EARLY_END")
T3_LATE=$(slice_median  "$T3_LIST" "$LATE_START"  "$LATE_END")
S3_EARLY=$(slice_median "$S3_LIST" "$EARLY_START" "$EARLY_END")
S3_LATE=$(slice_median  "$S3_LIST" "$LATE_START"  "$LATE_END")
T3_DRIFT_PCT=$(awk "BEGIN { printf \"%d\", ($T3_LATE - $T3_EARLY) * 100 / $T3_EARLY }")
S3_DRIFT_PCT=$(awk "BEGIN { printf \"%d\", ($S3_LATE - $S3_EARLY) * 100 / $S3_EARLY }")

echo "    T3   early=${T3_EARLY}ms late=${T3_LATE}ms  drift=${T3_DRIFT_PCT}%"
echo "    S3Gen early=${S3_EARLY}ms late=${S3_LATE}ms  drift=${S3_DRIFT_PCT}%"

abs() { local n="$1"; if [ "$n" -lt 0 ]; then echo $((-n)); else echo "$n"; fi; }
if [ "$(abs "$T3_DRIFT_PCT")" -gt 5 ]; then
    echo "    FAIL: T3 drift > 5 %" >&2
    exit 1
fi
if [ "$(abs "$S3_DRIFT_PCT")" -gt 5 ]; then
    echo "    FAIL: S3Gen drift > 5 %" >&2
    exit 1
fi
echo "    PASS: drift within ±5 %"

# ---------------------------------------------------------------------------
# 3. RSS growth
# ---------------------------------------------------------------------------
echo "==> 3/4  RSS growth"
RSS_EARLY=$(slice_median "$RSS_LIST" "$EARLY_START" "$EARLY_END")
RSS_LATE=$(slice_median  "$RSS_LIST" "$LATE_START"  "$LATE_END")
RSS_DELTA_KB=$(( RSS_LATE - RSS_EARLY ))
echo "    RSS early=${RSS_EARLY}KB late=${RSS_LATE}KB  delta=${RSS_DELTA_KB}KB"
if [ "$RSS_DELTA_KB" -gt 10240 ]; then
    echo "    FAIL: RSS grew by > 10 MB across runs" >&2
    exit 1
fi
echo "    PASS: RSS growth within 10 MB"

# ---------------------------------------------------------------------------
# 4. NV ComputeCache growth bounded
# ---------------------------------------------------------------------------
echo "==> 4/4  ~/.nv/ComputeCache growth"
echo "    NV cache: ${NV_BEFORE_KB}KB → ${NV_AFTER_KB}KB (delta=${NV_DELTA_KB}KB)"
if [ "$NV_DELTA_KB" -gt 51200 ]; then
    echo "    FAIL: NV ComputeCache grew by > 50 MB" >&2
    exit 1
fi
echo "    PASS: NV cache growth bounded"

echo
echo "All chatterbox.cpp CUDA stability tests PASSED ($N_RUNS runs)."
