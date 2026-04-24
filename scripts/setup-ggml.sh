#!/usr/bin/env bash
# Clone ggml into ./ggml, check out the commit this repo is pinned against,
# and apply every patch under ./patches/*.patch (Metal + OpenCL + Vulkan).
# Idempotent: safe to re-run.
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
    "ggml-opencl-chatterbox-ops.patch"
    "ggml-vulkan-pipeline-cache.patch"
)

echo "chatterbox.cpp: setting up ggml at pinned commit ${GGML_COMMIT}"

if [ ! -d ggml/.git ]; then
    echo "  → cloning ${GGML_URL}"
    git clone "$GGML_URL" ggml
fi

cd ggml

# Skip if we're already at the pinned commit with every patch already applied.
CURRENT="$(git rev-parse --short=8 HEAD 2>/dev/null || echo '')"
DIRTY_FILES="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
if [ "$CURRENT" = "$GGML_COMMIT" ] && [ "$DIRTY_FILES" -ge 1 ]; then
    ALL_APPLIED=1
    for p in "${PATCHES[@]}"; do
        # If the patch would apply cleanly on top, it isn't in yet.
        if git apply --check "$REPO_ROOT/patches/$p" 2>/dev/null; then
            ALL_APPLIED=0
            break
        fi
    done
    if [ "$ALL_APPLIED" = "1" ]; then
        echo "  → patches already applied on ${GGML_COMMIT}, nothing to do"
        exit 0
    fi
fi

echo "  → resetting to ${GGML_COMMIT} (discarding uncommitted changes under ./ggml)"
git fetch origin 2>/dev/null || true
git reset --hard "$GGML_COMMIT"
# Remove untracked files (e.g. left over from a previously applied patch) so
# reapply is deterministic; ggml/ is not intended for long-lived local work.
git clean -fdq

for p in "${PATCHES[@]}"; do
    echo "  → applying patches/$p"
    git apply "$REPO_ROOT/patches/$p"
done

N_METAL="$(git status --porcelain src/ggml-metal/ 2>/dev/null | wc -l | tr -d ' ')"
N_OPENCL="$(git status --porcelain include/ggml-opencl.h src/ggml-opencl/ 2>/dev/null | wc -l | tr -d ' ')"
N_VULKAN="$(git status --porcelain src/ggml-vulkan/ 2>/dev/null | wc -l | tr -d ' ')"
echo "  → ok (Metal: ${N_METAL}, OpenCL: ${N_OPENCL}, Vulkan: ${N_VULKAN} paths touched under ggml/)"
echo
echo "ggml is ready.  Next:"
echo "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON   # Apple"
echo "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENCL=ON  # Android / Termux"
echo "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON  # Linux/Windows/Android"
echo "    cmake --build build -j\$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
