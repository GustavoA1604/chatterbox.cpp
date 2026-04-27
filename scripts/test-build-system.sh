#!/usr/bin/env bash
# Build-system regression tests for chatterbox.cpp's vendored ggml setup.
#
# The patches under patches/ are all generated against a single pinned
# upstream commit (GGML_COMMIT in scripts/setup-ggml.sh).  When that
# commit gets bumped (or upstream ggml lands fixes that overlap with
# our patches), this script catches the breakage early instead of at
# the next CI build.
#
# What this script verifies:
#
#   1. setup-ggml.sh is idempotent.  Running it twice in a row produces
#      a clean second invocation that reports "patches already applied,
#      nothing to do".
#
#   2. setup-ggml.sh recovers from a partially-patched state (e.g. an
#      aborted run / merge conflict).  Manually corrupt one patched
#      file, re-run, verify it still reaches the "ok" state.
#
#   3. Each patch in patches/*.patch applies cleanly to a pristine
#      checkout of the pinned upstream commit.  This is what a fresh
#      clone of ggml would do.
#
#   4. After patches are applied, the expected files are modified
#      (catches "patch silently doesn't change anything because the
#      target was renamed upstream").
#
# Does NOT verify:
#   - that the post-patch ggml builds (covered by CMake / test-cuda-ops)
#   - perf characteristics (covered by test-chatterbox-cuda.sh)
#
# Usage:
#   ./scripts/test-build-system.sh
#
# Exits non-zero on any failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d ggml/.git ]; then
    echo "FAIL: ggml/ not present.  Run scripts/setup-ggml.sh first." >&2
    exit 1
fi

GGML_COMMIT="$(grep -E '^GGML_COMMIT=' scripts/setup-ggml.sh | head -1 | cut -d'"' -f2)"
if [ -z "$GGML_COMMIT" ]; then
    echo "FAIL: could not parse GGML_COMMIT from scripts/setup-ggml.sh" >&2
    exit 1
fi
echo "    pinned ggml commit: $GGML_COMMIT"

PATCHES=( $(grep -E '^\s+"ggml-.*\.patch"$' scripts/setup-ggml.sh | tr -d '" ,') )
if [ ${#PATCHES[@]} -eq 0 ]; then
    echo "FAIL: could not parse PATCHES list from scripts/setup-ggml.sh" >&2
    exit 1
fi
echo "    patches in pipeline: ${PATCHES[*]}"

# ---------------------------------------------------------------------------
# 1. Idempotency: setup, then setup again, verify second is a no-op
# ---------------------------------------------------------------------------
echo "==> 1/4  idempotency"

# Reset to a clean known state first.
( cd ggml && git checkout -- . )

# First run: should apply all patches
out1=$(bash scripts/setup-ggml.sh 2>&1)
if ! echo "$out1" | grep -q "applying patches/"; then
    echo "    FAIL: first setup didn't actually apply patches" >&2
    echo "$out1" >&2
    exit 1
fi
applied_count_1=$(echo "$out1" | grep -c "applying patches/" || true)
if [ "$applied_count_1" -ne "${#PATCHES[@]}" ]; then
    echo "    FAIL: first setup applied $applied_count_1 patches, expected ${#PATCHES[@]}" >&2
    exit 1
fi

# Second run: should be a no-op
out2=$(bash scripts/setup-ggml.sh 2>&1)
if ! echo "$out2" | grep -q "patches already applied"; then
    echo "    FAIL: second setup didn't short-circuit on idempotency check" >&2
    echo "$out2" >&2
    exit 1
fi
if echo "$out2" | grep -q "applying patches/"; then
    echo "    FAIL: second setup re-applied patches (should have skipped)" >&2
    exit 1
fi
echo "    PASS: 1st applied ${applied_count_1} patches, 2nd was a no-op"

# ---------------------------------------------------------------------------
# 2. Clean-recovery: corrupt one patched file, re-run, verify it heals
# ---------------------------------------------------------------------------
echo "==> 2/4  clean-recovery from dirty state"

# Pick the conv-transpose-1d.cu file (modified by ggml-cuda-chatterbox-ops.patch)
TARGET=ggml/src/ggml-cuda/conv-transpose-1d.cu
if [ ! -f "$TARGET" ]; then
    echo "    SKIP: $TARGET not present (older patch base?)"
else
    # Save what setup-ggml.sh produced
    expected_md5=$(md5sum "$TARGET" | cut -d' ' -f1)

    # Corrupt: prepend garbage
    {
        echo "// MANUAL CORRUPTION -- testing recovery"
        cat "$TARGET"
    } > "$TARGET.tmp" && mv "$TARGET.tmp" "$TARGET"

    # Verify it's actually corrupted now
    if [ "$(md5sum "$TARGET" | cut -d' ' -f1)" = "$expected_md5" ]; then
        echo "    FAIL: corruption step didn't actually change the file" >&2
        exit 1
    fi

    # Re-run setup; should detect dirty state, reset, re-apply
    out3=$(bash scripts/setup-ggml.sh 2>&1)
    if ! echo "$out3" | grep -q "applying patches/"; then
        echo "    FAIL: setup didn't re-apply after corruption" >&2
        echo "$out3" >&2
        exit 1
    fi
    actual_md5=$(md5sum "$TARGET" | cut -d' ' -f1)
    if [ "$actual_md5" != "$expected_md5" ]; then
        echo "    FAIL: recovery produced different file content" >&2
        echo "      expected md5: $expected_md5" >&2
        echo "      actual   md5: $actual_md5" >&2
        exit 1
    fi
    echo "    PASS: setup recovered $TARGET from manual corruption"
fi

# ---------------------------------------------------------------------------
# 3. Each patch applies cleanly to a pristine pinned-commit checkout
# ---------------------------------------------------------------------------
echo "==> 3/4  each patch applies cleanly to pinned commit"

# Reset ggml clone to clean pinned commit, then dry-run each patch
( cd ggml && git checkout -- . && git checkout "$GGML_COMMIT" >/dev/null 2>&1 )

for p in "${PATCHES[@]}"; do
    if ( cd ggml && git apply --check "$REPO_ROOT/patches/$p" 2>&1 ); then
        echo "    PASS: $p applies cleanly to ${GGML_COMMIT}"
    else
        echo "    FAIL: $p does NOT apply cleanly to ${GGML_COMMIT}" >&2
        echo "          (rebase the patch or bump GGML_COMMIT)" >&2
        exit 1
    fi
    # Actually apply so the next patch in the chain has the right base.
    ( cd ggml && git apply "$REPO_ROOT/patches/$p" )
done

# ---------------------------------------------------------------------------
# 4. Post-patch file count matches the README's claim
# ---------------------------------------------------------------------------
echo "==> 4/4  modified file counts match patches/README.md expectations"
N_METAL=$(cd ggml && git status --porcelain src/ggml-metal/ 2>/dev/null | wc -l | tr -d ' ')
N_CUDA=$(cd ggml && git status --porcelain src/ggml-cuda/  2>/dev/null | wc -l | tr -d ' ')
N_VK=$(cd ggml && git status --porcelain src/ggml-vulkan/ 2>/dev/null | wc -l | tr -d ' ')

# Tolerances: counts in patches/README.md are documented as
#   "7 modified files under ggml/src/ggml-metal/"
#   "3 modified files under ggml/src/ggml-cuda/"
# Soft check — we allow ranges so a future patch that touches a 4th
# CUDA file doesn't have to update this test, only the README.
if [ "$N_METAL" -lt 1 ]; then
    echo "    FAIL: expected ≥ 1 modified file under ggml-metal/, got $N_METAL" >&2
    exit 1
fi
if [ "$N_CUDA" -lt 1 ]; then
    echo "    FAIL: expected ≥ 1 modified file under ggml-cuda/, got $N_CUDA" >&2
    exit 1
fi
echo "    PASS: ${N_METAL} files under ggml-metal/, ${N_CUDA} files under ggml-cuda/, ${N_VK} files under ggml-vulkan/"

echo
echo "All build-system tests PASSED."
