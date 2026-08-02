"""Detects the system MPI implementation and exposes it as @mpi//:mpi.

Mirrors CMake's find_package(MPI): runs `mpicxx -show` to get the
real compile/link flags for whatever MPI is on PATH.
"""

def _mpi_configure_impl(rctx):
    mpicxx = rctx.os.environ.get("MPICXX", "mpicxx")
    show = rctx.execute([mpicxx, "-show"])
    if show.return_code != 0:
        fail((
            "Could not run '{} -show' to detect MPI (exit {}): {}\n" +
            "Make sure an MPI implementation is available on PATH " +
            "(e.g. `module load mpi` on an HPC cluster, or install " +
            "OpenMPI/MPICH locally) before running bazel build. Set " +
            "MPICXX to override which mpicxx-like wrapper is used."
        ).format(mpicxx, show.return_code, show.stderr))

    tokens = [t for t in show.stdout.strip().split(" ") if t]
    include_dir = None
    linkopts = []
    for tok in tokens[1:]:  # tokens[0] is the underlying compiler (gcc/g++/...)
        if tok.startswith("-I") and include_dir == None:
            include_dir = tok[len("-I"):]
        elif tok.startswith("-L") or tok.startswith("-l") or tok.startswith("-Wl"):
            linkopts.append(tok)

    if include_dir:
        cp = rctx.execute(["cp", "-rL", include_dir, "include"])
        if cp.return_code != 0:
            fail("Could not copy MPI include dir '{}' into the repo: {}".format(include_dir, cp.stderr))

    rctx.file("BUILD", """
cc_library(
    name = "mpi",
    hdrs = glob(["include/**"]),
    includes = ["include"],
    linkopts = {linkopts},
    visibility = ["//visibility:public"],
)
""".format(linkopts = repr(linkopts)))

mpi_configure = repository_rule(
    implementation = _mpi_configure_impl,
    environ = ["MPICXX", "PATH"],
    doc = "Detects the system MPI (mpicxx -show) and wraps it as a cc_library.",
)

def _mpi_extension_impl(_module_ctx):
    mpi_configure(name = "mpi")

mpi_extension = module_extension(
    implementation = _mpi_extension_impl,
    doc = "Registers @mpi//:mpi, a cc_library for the system MPI implementation.",
)
