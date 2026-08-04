"""Cross-process MPI_Allreduce via Client.compile_and_load().

Constructs the compile request directly (num_replicas=N, num_partitions=1)
against a StableHLO module.
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
module @verify_allreduce attributes {{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = {size} : i32}} {{
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {{
    %0 = "stablehlo.all_reduce"(%arg0) ({{
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
      %s = stablehlo.add %a, %b : tensor<f32>
      stablehlo.return %s : tensor<f32>
    }}) {{replica_groups = dense<[[{", ".join(str(i) for i in range(size))}]]> : tensor<1x{size}xi64>}} : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
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

local_data = jax.device_put(np.array(float(rank), dtype=np.float32), jax.devices()[rank])
result = executable.execute([local_data])
got = np.asarray(result[0])
expected = float(sum(range(size)))
print(f"Rank {rank}: result = {got}; expected = {expected}")
assert got == expected, f"Rank {rank}: expected {expected}, got {got}"
print(f"Rank {rank}: OK")

jax.distributed.shutdown()
