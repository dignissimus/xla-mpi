"""Cross-process MPI_Iallgather along a non-outermost dimension.

Each rank contributes a (2, 1) column. Gathering along dim 1 concatenates
each rank's column side by side, so every rank's result should be the same
(2, size) matrix.
"""

import numpy as np
import jax
from jax._src.lib import xla_client as xc

jax.config.update("jax_platforms", "mpi")
jax.distributed.initialize(cluster_detection_method="mpi4py")

rank = jax.process_index()
size = jax.process_count()

print(f"Rank {rank}: Devices = {jax.devices()}")

stablehlo_text = f"""
module @verify_all_gather_dim1 attributes {{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = {size} : i32}} {{
  func.func public @main(%arg0: tensor<2x1xf32>) -> tensor<2x{size}xf32> {{
    %0 = "stablehlo.all_gather"(%arg0) {{all_gather_dim = 1 : i64,
         replica_groups = dense<[[{", ".join(str(i) for i in range(size))}]]> : tensor<1x{size}xi64>}}
      : (tensor<2x1xf32>) -> tensor<2x{size}xf32>
    return %0 : tensor<2x{size}xf32>
  }}
}}
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

local_data = jax.device_put(
    np.array([[10.0 * rank + row] for row in range(2)], dtype=np.float32), jax.devices()[rank])
result = executable.execute([local_data])
got = np.asarray(result[0])
expected = np.array([[10.0 * k + row for k in range(size)] for row in range(2)], dtype=np.float32)
print(f"Rank {rank}: result = {got.tolist()}; expected = {expected.tolist()}")
assert np.array_equal(got, expected), f"Rank {rank}: expected {expected}, got {got}"
print(f"Rank {rank}: OK")

jax.distributed.shutdown()
