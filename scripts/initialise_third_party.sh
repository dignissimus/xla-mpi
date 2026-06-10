#!/usr/bin/env bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
THIRD_PARTY_DIR="${SCRIPT_DIR}/third_party"

DEST_DIR="${THIRD_PARTY_DIR}/xla/pjrt/c"
DEST_FILE="${DEST_DIR}/pjrt_c_api.h"
# main: HEADER_URL="https://raw.githubusercontent.com/openxla/xla/main/xla/pjrt/c/pjrt_c_api.h"
# 0.5.0 header
HEADER_URL="https://raw.githubusercontent.com/openxla/xla/0d1b60216ea13b0d261d59552a0f7ef20c4f76c5/xla/pjrt/c/pjrt_c_api.h"

echo "Creating directory structure at: third_party/xla/pjrt/c/"
mkdir -p "$DEST_DIR"

echo "Downloading pjrt_c_api.h from OpenXLA branch..."
curl -# -L -o "$DEST_FILE" "$HEADER_URL"

echo "The PJRT C API header is now vendored at: $DEST_FILE"
