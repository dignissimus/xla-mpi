module load mpi
export PJRT_NAMES_AND_LIBRARY_PATHS="mpi:$(pwd)/build/lib/libpjrt_plugin_mpi.so" 
. .venv/bin/activate
