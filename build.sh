#!/usr/bin/env bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "Building xla-mpi"

if [ -d "$BUILD_DIR" ]; then
    echo "Found existing build directory. Cleaning up..."
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring project with CMake..."
STABLEHLO_FLAG=""
if [ -n "$STABLEHLO_DIR" ]; then
    STABLEHLO_FLAG="-DStablehlo_DIR=$STABLEHLO_DIR"
fi
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    $STABLEHLO_FLAG

echo "Compiling shared library..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    NUM_CORES=$(sysctl -n hw.ncpu)
else
    NUM_CORES=$(nproc)
fi

cmake --build . --config Release --parallel "$NUM_CORES"

echo "Build Successful!"
echo "The plugin is located at: ${BUILD_DIR}/lib/libpjrt_plugin_mpi.so" 
echo "(Or .dylib if you are building on macOS)"
