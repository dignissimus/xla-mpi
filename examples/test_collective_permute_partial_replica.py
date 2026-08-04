"""stablehlo.collective_permute where not every rank is a target.

Requires -np 4. Pairs form a chain [0->1, 1->2] rather than covering every
rank: rank 0 is a source only, rank 3 is neither a source nor a target.
StableHLO's collective_permute specification says that any device that is not a target of
some pair gets a zero-valued result.
"""

import numpy as np
import jax
from jax._src.lib import xla_client as xc

jax.config.update("jax_platforms", "mpi")
jax.distributed.initialize(cluster_detection_method="mpi4py")

rank = jax.process_index()
size = jax.process_count()
assert size == 4, "test_collective_permute_partial_replica.py requires -np 4"

print(f"Rank {rank}: Devices = {jax.devices()}")

stablehlo_text = """
module @verify_collective_permute_partial attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 4 : i32} {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.collective_permute"(%arg0) {source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>}
      : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}
"""

c = jax.devices()[0].client

opts = xc.CompileOptions()
opts.num_replicas = size
opts.num_partitions = 1
assignment = np.array([[jax.devices()[i].id] for i in range(size)], dtype=np.int64)
da = xc.DeviceAssignment.create(assignment)
opts.device_assignment = da
opts.executable_build_options.num_replicas = size
opts.executable_build_options.num_partitions = 1
opts.executable_build_options.device_assignment = da

executable = c.compile_and_load(stablehlo_text, jax.devices(), opts)
print(f"Rank {rank}: compiled OK")

local_data = jax.device_put(np.array(100.0 + rank, dtype=np.float32), jax.devices()[rank])
result = executable.execute([local_data])
got = np.asarray(result[0])
# rank 0: source only, never a target -> zero-fills.
# rank 1: target of (0,1) -> gets rank 0's value (100.0).
# rank 2: target of (1,2) -> gets rank 1's value (101.0).
# rank 3: neither source nor target of any pair -> zero-fills.
expected = {0: 0.0, 1: 100.0, 2: 101.0, 3: 0.0}[rank]
print(f"Rank {rank}: result = {got}; expected = {expected}")
assert got == expected, f"Rank {rank}: expected {expected}, got {got}"
print(f"Rank {rank}: OK")

jax.distributed.shutdown()
