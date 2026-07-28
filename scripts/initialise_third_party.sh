#!/usr/bin/env bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd )"
THIRD_PARTY_DIR="${REPO_ROOT}/third_party"

DEST_DIR="${THIRD_PARTY_DIR}/xla/pjrt/c"
DEST_FILE="${DEST_DIR}/pjrt_c_api.h"
# main: HEADER_URL="https://raw.githubusercontent.com/openxla/xla/main/xla/pjrt/c/pjrt_c_api.h"
# XLA commit from jax-v0.11.0
HEADER_URL="https://raw.githubusercontent.com/openxla/xla/131bf41acb4650e4391a640c3f1859c1c86ad74b/xla/pjrt/c/pjrt_c_api.h"

echo "Creating directory structure at: third_party/xla/pjrt/c/"
mkdir -p "$DEST_DIR"

echo "Downloading pjrt_c_api.h from OpenXLA branch..."
curl -# -L -o "$DEST_FILE" "$HEADER_URL"

echo "The PJRT C API header is now vendored at: $DEST_FILE"
