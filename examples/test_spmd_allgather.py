"""Cross-process SPMD + cross-partition AllGather.


Each rank contributes its own partition_id-independent constant.
Gathering across all `size` partitions must produce the same vector on every rank.
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
module @verify_spmd_allgather attributes {{mhlo.num_partitions = {size} : i32, mhlo.num_replicas = 1 : i32}} {{
  func.func public @main(%arg0: tensor<1xf32>) -> tensor<{size}xf32> {{
    %0 = "stablehlo.all_gather"(%arg0) {{
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[{", ".join(str(i) for i in range(size))}]]> : tensor<1x{size}xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
      use_global_device_ids
    }} : (tensor<1xf32>) -> tensor<{size}xf32>
    return %0 : tensor<{size}xf32>
  }}
}}
"""

c = jax.devices()[0].client

opts = xc.CompileOptions()
opts.num_replicas = 1
opts.num_partitions = size
assignment = np.array([[jax.devices()[i].id for i in range(size)]], dtype=np.int64)
da = xc.DeviceAssignment.create(assignment)
opts.device_assignment = da
opts.executable_build_options.num_replicas = 1
opts.executable_build_options.num_partitions = size
opts.executable_build_options.device_assignment = da
opts.executable_build_options.use_spmd_partitioning = True

executable = c.compile_and_load(stablehlo_text, jax.devices(), opts)
print(f"Rank {rank}: compiled OK")

local_data = jax.device_put(np.array([100.0 + rank], dtype=np.float32), jax.devices()[rank])
result = executable.execute([local_data])
got = np.asarray(result[0])
expected = np.array([100.0 + i for i in range(size)], dtype=np.float32)
print(f"Rank {rank}: result = {got}; expected = {expected}")
assert np.array_equal(got, expected), f"Rank {rank}: expected {expected}, got {got}"
print(f"Rank {rank}: OK")

jax.distributed.shutdown()
