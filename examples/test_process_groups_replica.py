"""Cross-process MPI_Allreduce restricted to sub-groups, via Client.compile_and_load().

Tests partitioning ranks into independent subgroups. Splits into two groups
of 2: [0, 1] and [2, 3]. Requires -np 4.
"""

import numpy as np
import jax
from jax._src.lib import xla_client as xc

jax.config.update("jax_platforms", "mpi")
jax.distributed.initialize(cluster_detection_method="mpi4py")

rank = jax.process_index()
size = jax.process_count()
assert size == 4, "test_process_groups_replica.py requires -np 4"

print(f"Rank {rank}: Devices = {jax.devices()}")

stablehlo_text = """
module @verify_process_groups attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 4 : i32} {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
      %s = stablehlo.add %a, %b : tensor<f32>
      stablehlo.return %s : tensor<f32>
    }) {replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>} : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}
"""

c = jax.devices()[0].client

opts = xc.CompileOptions()
opts.num_replicas = size
opts.num_partitions = 1
assignment = np.array([[jax.devices()[i].id] for i in range(size)], dtype=np.int64)  # real packed GlobalDeviceId, not raw rank i
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
expected = 201.0 if rank in (0, 1) else 205.0
print(f"Rank {rank}: result = {got}; expected = {expected}")
assert got == expected, f"Rank {rank}: expected {expected}, got {got}"
print(f"Rank {rank}: OK")

jax.distributed.shutdown()
