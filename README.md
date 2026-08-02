# xla-mpi

PJRT plugin adding MPI as an XLA Backend

## Building

Builds with Bazel

```bash
./build.sh
```

You'll also need an MPI implementation available on `PATH` before running `bazel build`: e.g. `module load mpi` on an HPC cluster, or a local OpenMPI/MPICH install.

