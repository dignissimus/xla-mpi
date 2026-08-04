#!/bin/bash
set -u
cd "$(dirname "$0")"

export PJRT_NAMES_AND_LIBRARY_PATHS="mpi:$(pwd)/bazel-bin/libpjrt_plugin_mpi.so"
source /nobackup/vqqw43/xla-mpi-phase2-venv/bin/activate

fail=0
run() {
    local np="$1"; shift
    local test="$1"; shift
    echo "=== -np $np $test ==="
    if mpirun -np "$np" python "$test" > "/tmp/regress_$(basename "$test" .py)_np${np}.log" 2>&1; then
        echo "PASS"
    else
        echo "FAIL"
        tail -40 "/tmp/regress_$(basename "$test" .py)_np${np}.log"
        fail=$((fail + 1))
    fi
}

for t in examples/test_allreduce_replica.py examples/test_allreduce_max_replica.py \
         examples/test_allreduce_min_replica.py examples/test_allreduce_product_replica.py \
         examples/test_reduce_scatter_replica.py examples/test_reduce_scatter_dim1_replica.py \
         examples/test_allgather_replica.py examples/test_allgather_dim1_replica.py \
         examples/test_all_to_all_replica.py examples/test_collective_permute_replica.py; do
    run 2 "$t"
    run 4 "$t"
done

# Tests below assert a specific rank count internally and only make sense at that count.
run 2 examples/test_send_recv_replica.py
run 2 examples/test_data_parallel_grad_replica.py
run 2 examples/test_nbody_pmap.py
run 2 examples/test_transformer_block_shard_map.py

run 4 examples/test_collective_permute_partial_replica.py
run 4 examples/test_send_recv_chain_replica.py
run 4 examples/test_process_groups_replica.py
run 4 examples/test_spmd_allreduce.py
run 4 examples/test_spmd_allgather.py
run 4 examples/test_shard_map_psum.py

echo "SUMMARY: fail=$fail"
