#!/usr/bin/env bash

set -e

echo "Building xla-mpi"

if [[ "$OSTYPE" == "darwin"* ]]; then
    NUM_CORES=$(sysctl -n hw.ncpu)
else
    NUM_CORES=$(nproc)
fi

bazel build --jobs="$NUM_CORES" //:libpjrt_plugin_mpi.so

echo "Build Successful!"
echo "The plugin is located at: bazel-bin/libpjrt_plugin_mpi.so"
