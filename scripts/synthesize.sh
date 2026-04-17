#!/usr/bin/env bash
# End-to-end text -> wav synthesis using the C++/ggml pipeline.
#
# Pipeline:
#   chatterbox   (T3)     : text           -> speech_tokens
#   chatterbox-tts        : speech_tokens  -> wav (S3Gen + HiFT)
#
# Usage:
#   scripts/synthesize.sh "Hello, world." out.wav
#   scripts/synthesize.sh "Hello, world." out.wav --seed 123

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 TEXT OUT.wav [--seed N]" >&2
    exit 1
fi

TEXT="$1"
OUT="$2"
shift 2
EXTRA_ARGS="$*"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
T3_BIN="$ROOT/build/chatterbox"
TTS_BIN="$ROOT/build/chatterbox-tts"
T3_GGUF="$ROOT/models/chatterbox-t3-turbo.gguf"
S3G_GGUF="$ROOT/models/chatterbox-s3gen.gguf"

# Tokenizer lookup order:
#   1. $CHATTERBOX_TOKENIZER_DIR if set
#   2. <repo>/tokenizer/ if it exists
#   3. HF snapshot dir if present
if [[ -n "${CHATTERBOX_TOKENIZER_DIR:-}" ]] && [[ -f "$CHATTERBOX_TOKENIZER_DIR/vocab.json" ]]; then
    TOKENIZER_DIR="$CHATTERBOX_TOKENIZER_DIR"
elif [[ -f "$ROOT/tokenizer/vocab.json" ]]; then
    TOKENIZER_DIR="$ROOT/tokenizer"
else
    HF_SNAPSHOT_DIR="$(ls -d $HOME/.cache/huggingface/hub/models--ResembleAI--chatterbox-turbo/snapshots/*/ 2>/dev/null | head -1 || true)"
    if [[ -n "$HF_SNAPSHOT_DIR" ]] && [[ -f "$HF_SNAPSHOT_DIR/vocab.json" ]]; then
        TOKENIZER_DIR="$HF_SNAPSHOT_DIR"
    else
        echo "error: could not locate vocab.json / merges.txt. Set" >&2
        echo "       CHATTERBOX_TOKENIZER_DIR, or place them in $ROOT/tokenizer/" >&2
        exit 1
    fi
fi

if [[ ! -x "$T3_BIN" ]] || [[ ! -x "$TTS_BIN" ]]; then
    echo "error: binaries not built; run 'cmake --build build' first" >&2
    exit 1
fi
for f in "$T3_GGUF" "$S3G_GGUF"; do
    [[ -f "$f" ]] || { echo "error: missing $f" >&2; exit 1; }
done

TMP="$(mktemp)"
trap "rm -f $TMP" EXIT

echo ">>> [1/2] T3: text -> speech tokens (tokenizer: $TOKENIZER_DIR)"
"$T3_BIN" \
    --model "$T3_GGUF" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --text "$TEXT" \
    --output "$TMP" \
    ${EXTRA_ARGS} > /dev/null

N_TOK=$(tr ',' '\n' < "$TMP" | wc -l | tr -d ' ')
echo "    generated $N_TOK speech tokens"

echo ">>> [2/2] S3Gen + HiFT: speech tokens -> wav (built-in voice from $S3G_GGUF)"
"$TTS_BIN" \
    --s3gen-gguf "$S3G_GGUF" \
    --tokens-file "$TMP" \
    --out "$OUT" \
    ${EXTRA_ARGS}

echo "done: $OUT"
