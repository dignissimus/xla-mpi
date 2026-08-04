"""Cross-process all_to_all via MPI_Type_create_subarray.

Tests stablehlo.all_to_all with (num_replicas=N, num_partitions=1) using a
StableHLO module, with the Client.compile_and_load() API.
Uses split_dimension = concat_dimension = 0.

Each rank r's input is a `size`-length vector.
The k-th chunk of rank r's data is sent to rank k and placed at position r
in the concatenated output, so rank j's result[i] should be rank i's j-th
input element.
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
module @verify_all_to_all attributes {{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = {size} : i32}} {{
  func.func public @main(%arg0: tensor<{size}xf32>) -> tensor<{size}xf32> {{
    %0 = "stablehlo.all_to_all"(%arg0) {{split_dimension = 0 : i64,
         concat_dimension = 0 : i64,
         split_count = {size} : i64,
         replica_groups = dense<[[{", ".join(str(i) for i in range(size))}]]> : tensor<1x{size}xi64>}}
      : (tensor<{size}xf32>) -> tensor<{size}xf32>
    return %0 : tensor<{size}xf32>
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
    np.array([100.0 * rank + k for k in range(size)], dtype=np.float32), jax.devices()[rank])
result = executable.execute([local_data])
got = np.asarray(result[0])
expected = np.array([100.0 * i + rank for i in range(size)], dtype=np.float32)
print(f"Rank {rank}: result = {got}; expected = {expected}")
assert np.array_equal(got, expected), f"Rank {rank}: expected {expected}, got {got}"
print(f"Rank {rank}: OK")

jax.distributed.shutdown()
