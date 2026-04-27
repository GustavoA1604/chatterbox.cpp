#!/usr/bin/env bash
# Clone ggml into ./ggml, check out the commit this repo is pinned against,
# and apply every patch under ./patches/*.patch.  Idempotent: safe to re-run.
#
# Update GGML_COMMIT here whenever any patch is re-generated against a newer
# upstream ggml; this file is the single source of truth for the pin.

set -euo pipefail

# -----------------------------------------------------------------------------
# The upstream ggml commit all patches under ./patches/ were authored
# against.  Pin here so fresh clones (and CI) build deterministically.
# -----------------------------------------------------------------------------
GGML_COMMIT="58c38058"
GGML_URL="https://github.com/ggml-org/ggml.git"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# List of patches to apply, in order.  Keep in lock-step with patches/README.md.
PATCHES=(
    "ggml-metal-chatterbox-ops.patch"
    "ggml-cuda-chatterbox-ops.patch"
)

echo "chatterbox.cpp: setting up ggml at pinned commit ${GGML_COMMIT}"

if [ ! -d ggml/.git ]; then
    echo "  → cloning ${GGML_URL}"
    git clone "$GGML_URL" ggml
fi

cd ggml

# Skip if we're already at the pinned commit with every patch already applied.
#
# Use `git apply --reverse --check`: it asks "would the reverse of this
# patch apply cleanly?", which is true ONLY when the patch's exact
# expected-output content is currently in the tree.  This is much more
# discriminating than plain `--check` (which can spuriously fail —
# and so spuriously declare "already applied" — when the working tree
# is dirty in unrelated ways, e.g. an aborted previous run or manual
# debug edits).  See scripts/test-build-system.sh §2 for the recovery
# case this guards.
CURRENT="$(git rev-parse --short=8 HEAD 2>/dev/null || echo '')"
DIRTY_FILES="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
if [ "$CURRENT" = "$GGML_COMMIT" ] && [ "$DIRTY_FILES" -ge 1 ]; then
    ALL_APPLIED=1
    for p in "${PATCHES[@]}"; do
        # If the reverse-apply does NOT apply cleanly, this patch's
        # exact output is not in the tree — fall through and re-apply
        # everything from scratch.
        if ! git apply --reverse --check "$REPO_ROOT/patches/$p" 2>/dev/null; then
            ALL_APPLIED=0
            break
        fi
    done
    if [ "$ALL_APPLIED" = "1" ]; then
        echo "  → patches already applied on ${GGML_COMMIT}, nothing to do"
        exit 0
    fi
fi

echo "  → checking out ${GGML_COMMIT}"
# Reset any prior partial state first so `git apply` doesn't trip over
# stale diffs from an aborted run.
git checkout -- . 2>/dev/null || true
git checkout "$GGML_COMMIT"

for p in "${PATCHES[@]}"; do
    echo "  → applying patches/$p"
    git apply "$REPO_ROOT/patches/$p"
done

N_METAL="$(git status --porcelain src/ggml-metal/ 2>/dev/null | wc -l | tr -d ' ')"
N_CUDA="$(git status --porcelain src/ggml-cuda/  2>/dev/null | wc -l | tr -d ' ')"
echo "  → ok (${N_METAL} files modified under src/ggml-metal/, ${N_CUDA} under src/ggml-cuda/)"
echo
echo "ggml is ready.  Next:"
echo "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON   # Apple"
echo "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON    # Linux/Windows + NVIDIA"
echo "    cmake --build build -j\$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
