#!/bin/bash
# Common setup for xla-mpi dependency build scripts. Source, don't execute
# directly. Handles: CLI arg parsing, required-tool checks, the ninja
# version gate, and picking one consistent CC/CXX for every build below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEPS_DIR="${DEPS_DIR:-$HOME/.local/xla-mpi-deps}"
JOBS="${JOBS:-$( (command -v nproc >/dev/null 2>&1 && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
FORCE_REBUILD=false
ENV_FILE_OUT=""

print_common_help() {
    cat <<EOF
Options:
  --deps-dir PATH   Where to clone/build LLVM+MLIR and StableHLO
                     (default: \$DEPS_DIR or \$HOME/.local/xla-mpi-deps)
  --jobs N          Parallel build jobs (default: \$JOBS or nproc)
  --force           Rebuild even if the pinned commits already match
  --env-file PATH   Where to write the sourceable env file
                     (default: \$DEPS_DIR/env.sh)
  -h, --help        Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --deps-dir) DEPS_DIR="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --force) FORCE_REBUILD=true; shift ;;
        --env-file) ENV_FILE_OUT="$2"; shift 2 ;;
        -h|--help) print_common_help; exit 0 ;;
        *) echo "Unknown option: $1" >&2; print_common_help >&2; exit 1 ;;
    esac
done

ENV_FILE_OUT="${ENV_FILE_OUT:-$DEPS_DIR/env.sh}"

mkdir -p "$DEPS_DIR"

for tool in cmake git python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Error: '$tool' is required but was not found on PATH." >&2
        exit 1
    fi
done

NINJA="${NINJA:-ninja}"
if ! command -v "$NINJA" >/dev/null 2>&1; then
    echo "Error: ninja ('$NINJA') not found on PATH." >&2
    echo "Install a recent ninja (>= 1.10) or set NINJA=/path/to/ninja." >&2
    exit 1
fi
NINJA_VERSION="$("$NINJA" --version)"
NINJA_MAJOR="${NINJA_VERSION%%.*}"
NINJA_MINOR="$(echo "$NINJA_VERSION" | awk -F. '{print $2}')"
NINJA_TOO_OLD=false
if ! [ "$NINJA_MAJOR" -eq "$NINJA_MAJOR" ] 2>/dev/null; then
    NINJA_TOO_OLD=true  # unparseable version string; don't trust it
elif [ "$NINJA_MAJOR" -lt 1 ]; then
    NINJA_TOO_OLD=true
elif [ "$NINJA_MAJOR" -eq 1 ] && [ "$NINJA_MINOR" -lt 10 ] 2>/dev/null; then
    NINJA_TOO_OLD=true
fi
if [ "$NINJA_TOO_OLD" = true ]; then
    echo "Error: '$(command -v "$NINJA")' is ninja $NINJA_VERSION, but >= 1.10 is required." >&2
    echo "A stale ninja (seen as old as 1.8.2 on some HPC images) cannot parse" >&2
    echo "the build.ninja files modern LLVM/MLIR generates and fails with:" >&2
    echo "  ninja: error: ... multiple outputs aren't (yet?) supported by depslog" >&2
    echo "Fix: install/module-load a newer ninja, then either put it first on" >&2
    echo "PATH or set NINJA=/path/to/newer/ninja and re-run." >&2
    exit 1
fi

NINJA="$(command -v "$NINJA")"
echo "Using ninja $NINJA_VERSION ($NINJA)"

CC="${CC:-cc}"
CXX="${CXX:-c++}"
if ! CC_RESOLVED="$(command -v "$CC")"; then
    echo "Error: C compiler '$CC' (from \$CC) not found on PATH." >&2
    exit 1
fi
if ! CXX_RESOLVED="$(command -v "$CXX")"; then
    echo "Error: C++ compiler '$CXX' (from \$CXX) not found on PATH." >&2
    exit 1
fi
CC="$CC_RESOLVED"
CXX="$CXX_RESOLVED"
echo "Using CC=$CC"
echo "Using CXX=$CXX"
echo "(If your system needs a non-default compiler, e.g. via 'module load"
echo " gcc/...', export CC/CXX to its full path before running this script."
echo " The same CC/CXX is used for every dependency build below, on purpose.)"

COMPILER_LIB64=""
case "$CXX" in
    /*)
        _cxx_prefix="$(dirname "$(dirname "$CXX")")"
        if [ -d "$_cxx_prefix/lib64" ]; then
            COMPILER_LIB64="$_cxx_prefix/lib64"
        fi
        ;;
esac
if [ -n "$COMPILER_LIB64" ]; then
    echo "Compiler runtime lib dir for build-time tools: $COMPILER_LIB64"
fi
