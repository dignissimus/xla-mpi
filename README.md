# xla-mpi

PJRT plugin adding MPI as an XLA Backend

## Building

Needs LLVM/MLIR and StableHLO and needs the following environment variable set: `STABLEHLO_DIR`, `MLIR_DIR`, `LLVM_DIR`.
`scripts/setup_deps.sh` clones and builds these then prints the env vars for you.

```bash
./scripts/initialise_third_party.sh
./scripts/setup_deps.sh
. "$HOME/.local/xla-mpi-deps/env.sh"
./build.sh
```

You'll also need an MPI implementation available, e.g. `module load mpi` on an HPC cluster, or a local OpenMPI/MPICH install.

