#!/bin/bash
# Clone and build LLVM/MLIR + StableHLO, in place, for xla-mpi.
#
# Usage:
#   ./scripts/setup_deps_llvm.sh [--deps-dir DIR] [--jobs N] [--force] [--env-file PATH]
#
# Env overrides:
#   LLVM_COMMIT, STABLEHLO_COMMIT   override the pinned commits below
#   CC, CXX                        compiler used for BOTH LLVM and StableHLO
#   NINJA                          path to a specific ninja binary

set -euo pipefail

# shellcheck source=setup_deps_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup_deps_common.sh" "$@"

# Confirmed to build together and work with xla-mpi as of 2026-07-28,
LLVM_COMMIT="${LLVM_COMMIT:-b713aaedb6eedd7d20f666038d945e50eafd4c27}"
STABLEHLO_COMMIT="${STABLEHLO_COMMIT:-402df0aca1e420c4c6186a1444f0516d2934b6e1}"

LLVM_SRC_DIR="$DEPS_DIR/llvm-project"
LLVM_BUILD_DIR="$LLVM_SRC_DIR/build"
STABLEHLO_DIR="$DEPS_DIR/stablehlo"

STABLEHLO_BUILD_DIR="$STABLEHLO_DIR/build"

echo "=== xla-mpi native dependency setup (LLVM/MLIR + StableHLO) ==="
echo "Deps dir:  $DEPS_DIR"
echo "Jobs:      $JOBS"
echo "LLVM:      $LLVM_COMMIT"
echo "StableHLO: $STABLEHLO_COMMIT"
echo ""

STAMP_FILE="$DEPS_DIR/.llvm-versions"
EXPECTED_STAMP="llvm=$LLVM_COMMIT stablehlo=$STABLEHLO_COMMIT cc=$CC cxx=$CXX"
if [ -f "$STAMP_FILE" ] && [ "$(cat "$STAMP_FILE")" != "$EXPECTED_STAMP" ]; then
    echo "=== Pinned commit or compiler changed since last successful setup, forcing rebuild ==="
    FORCE_REBUILD=true
fi

if [ "$FORCE_REBUILD" = true ]; then
    echo "=== Rebuild requested: clearing previous build trees ==="
    rm -rf "$LLVM_BUILD_DIR" "$STABLEHLO_BUILD_DIR"
    rm -f "$STAMP_FILE"
fi

clone_pinned() {
    local dir="$1" url="$2" commit="$3"
    if [ ! -d "$dir/.git" ]; then
        echo "=== Fetching $(basename "$dir") @ $commit (shallow) ==="
        mkdir -p "$dir"
        (
            cd "$dir"
            git init -q
            git remote add origin "$url"
            git fetch --depth 1 origin "$commit"
            git checkout -q FETCH_HEAD
        )
    else
        local current
        current="$(git -C "$dir" rev-parse HEAD)"
        if [ "$current" != "$commit" ]; then
            echo "=== Updating $(basename "$dir") to $commit ==="
            (cd "$dir" && git fetch --depth 1 origin "$commit" && git checkout -q FETCH_HEAD)
        else
            echo "=== $(basename "$dir") already at $commit ==="
        fi
    fi
}

clone_pinned "$LLVM_SRC_DIR" "https://github.com/llvm/llvm-project.git" "$LLVM_COMMIT"
clone_pinned "$STABLEHLO_DIR" "https://github.com/openxla/stablehlo.git" "$STABLEHLO_COMMIT"

LLVM_BUILD_MARKER="$LLVM_BUILD_DIR/.xla-mpi-build-complete"
if [ ! -f "$LLVM_BUILD_MARKER" ]; then
    echo "=== Configuring LLVM/MLIR ==="
    cmake -G Ninja -B "$LLVM_BUILD_DIR" -S "$LLVM_SRC_DIR/llvm" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_MAKE_PROGRAM="$NINJA" \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_TARGETS_TO_BUILD=host \
        -DLLVM_ENABLE_PIC=ON \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DLLVM_ENABLE_ZSTD=OFF \
        -DLLVM_ENABLE_ZLIB=OFF \
        -DLLVM_ENABLE_BACKTRACES=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DLLVM_INCLUDE_DOCS=OFF \
        -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
        -DMLIR_ENABLE_EXECUTION_ENGINE=OFF

    echo "=== Building LLVM/MLIR (this is the slow part; expect 30-90+ minutes) ==="
    cmake --build "$LLVM_BUILD_DIR" --parallel "$JOBS"
    touch "$LLVM_BUILD_MARKER"
    echo "LLVM/MLIR built at $LLVM_BUILD_DIR"
else
    echo "=== LLVM/MLIR already built (use --force to rebuild) ==="
fi

# --- Patch StableHLO's test CMakeLists to tolerate a tests-less LLVM --------
# Guards each FileCheck/not reference with `if(TARGET FileCheck)` so the
# check-* lit targets are skipped instead of erroring at configure time.
# Only affects test-only targets, not the dialect libraries xla-mpi links.
patch_stablehlo_tests() {
    local files=(
        "$STABLEHLO_DIR/stablehlo/tests/CMakeLists.txt"
        "$STABLEHLO_DIR/stablehlo/testdata/CMakeLists.txt"
        "$STABLEHLO_DIR/stablehlo/conversions/linalg/tests/CMakeLists.txt"
        "$STABLEHLO_DIR/stablehlo/conversions/tosa/tests/CMakeLists.txt"
    )
    local f
    for f in "${files[@]}"; do
        if [ -f "$f" ] && ! grep -q "if(TARGET FileCheck)" "$f"; then
            python3 - "$f" <<'PYEOF'
import re
import sys

path = sys.argv[1]
content = open(path).read()
pattern = (
    r'(configure_lit_site_cfg\([^)]+\)\s*'
    r'add_lit_testsuite\([^)]+\)\s*'
    r'add_dependencies\([^)]+\))'
)
def wrap(m):
    return "if(TARGET FileCheck)\n" + m.group(1) + "\nendif()"

new_content, n = re.subn(pattern, wrap, content, count=1, flags=re.DOTALL)
if n == 0:
    print(
        f"WARNING: expected lit-testsuite pattern not found in {path}; "
        "StableHLO's CMakeLists.txt layout may have changed at this pin. "
        "Configure may fail below -- inspect this file manually.",
        file=sys.stderr,
    )
else:
    open(path, "w").write(new_content)
PYEOF
        fi
    done
}
patch_stablehlo_tests

STABLEHLO_BUILD_MARKER="$STABLEHLO_BUILD_DIR/.xla-mpi-build-complete"
if [ ! -f "$STABLEHLO_BUILD_MARKER" ]; then
    echo "=== Configuring StableHLO ==="
    cmake -G Ninja -B "$STABLEHLO_BUILD_DIR" -S "$STABLEHLO_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_MAKE_PROGRAM="$NINJA" \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
        -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
        -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF

    echo "=== Building StableHLO ==="
    if [ -n "$COMPILER_LIB64" ]; then
        LD_LIBRARY_PATH="$COMPILER_LIB64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            cmake --build "$STABLEHLO_BUILD_DIR" --parallel "$JOBS"
    else
        cmake --build "$STABLEHLO_BUILD_DIR" --parallel "$JOBS"
    fi
    touch "$STABLEHLO_BUILD_MARKER"
    echo "StableHLO built at $STABLEHLO_BUILD_DIR"
else
    echo "=== StableHLO already built (use --force to rebuild) ==="
fi

echo "$EXPECTED_STAMP" > "$STAMP_FILE"

MLIR_DIR_OUT="$LLVM_BUILD_DIR/lib/cmake/mlir"
LLVM_DIR_OUT="$LLVM_BUILD_DIR/lib/cmake/llvm"

mkdir -p "$(dirname "$ENV_FILE_OUT")"
cat > "$ENV_FILE_OUT" <<EOF
# Generated by scripts/setup_deps_llvm.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ).
# Source this before running ./build.sh:
#   . "$ENV_FILE_OUT"
export STABLEHLO_DIR="$STABLEHLO_DIR"
export MLIR_DIR="$MLIR_DIR_OUT"
export LLVM_DIR="$LLVM_DIR_OUT"
EOF

cat <<EOF

=== xla-mpi dependency setup complete ===

Export these before running ./build.sh (also written to $ENV_FILE_OUT):

  export STABLEHLO_DIR="$STABLEHLO_DIR"
  export MLIR_DIR="$MLIR_DIR_OUT"
  export LLVM_DIR="$LLVM_DIR_OUT"

  . "$ENV_FILE_OUT"   # or just source the file above

Note: build.sh also requires an MPI implementation to be available
(find_package(MPI) in CMakeLists.txt, e.g. \`module load mpi\` on an HPC
cluster, or an OpenMPI/MPICH install locally) -- that's independent of this
script and not something it sets up.

If you used a non-default CC/CXX for this setup, export the same CC/CXX
before running ./build.sh too, so the final plugin is linked with a
consistent toolchain against these static libraries.
EOF
