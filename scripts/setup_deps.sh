#!/bin/bash
# Builds LLVM/MLIR + StableHLO from source and prints the
# STABLEHLO_DIR/MLIR_DIR/LLVM_DIR env vars ./build.sh expects.
#
#   ./scripts/setup_deps.sh
#   . "$HOME/.local/xla-mpi-deps/env.sh"
#   ./build.sh
#
# Thin wrapper around setup_deps_llvm.sh; kept separate in case more
# dependencies get their own setup_deps_<name>.sh later.
#
# Options (forwarded to setup_deps_llvm.sh): --deps-dir PATH, --jobs N,
# --force, --env-file PATH. Env overrides: LLVM_COMMIT, STABLEHLO_COMMIT,
# CC, CXX, NINJA, DEPS_DIR, JOBS -- see setup_deps_llvm.sh/setup_deps_common.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/setup_deps_llvm.sh" "$@"
