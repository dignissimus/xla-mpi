"""Cross-process MPI_Allreduce with a MIN reduction.

Tests stablehlo.all_reduce with (num_replicas=N, num_partitions=1) using a
StableHLO module, with the Client.compile_and_load() API.
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
module @verify_allreduce_min attributes {{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = {size} : i32}} {{
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {{
    %0 = "stablehlo.all_reduce"(%arg0) ({{
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
      %s = stablehlo.minimum %a, %b : tensor<f32>
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
assignment = np.array([[jax.devices()[i].id] for i in range(size)], dtype=np.int64)
da = xc.DeviceAssignment.create(assignment)
opts.device_assignment = da
opts.executable_build_options.num_replicas = size
opts.executable_build_options.num_partitions = 1
opts.executable_build_options.device_assignment = da

executable = c.compile_and_load(stablehlo_text, jax.devices(), opts)
print(f"Rank {rank}: compiled OK")

# size - rank (descending, so rank 0 does NOT hold the minimum): a broken
# reduction that silently fell back to "first contribution" or "last
# contribution" would give the wrong answer instead of coincidentally
# matching.
local_data = jax.device_put(np.array(float(size - rank), dtype=np.float32), jax.devices()[rank])
result = executable.execute([local_data])
got = np.asarray(result[0])
expected = 1.0
print(f"Rank {rank}: result = {got}; expected = {expected}")
assert got == expected, f"Rank {rank}: expected {expected}, got {got}"
print(f"Rank {rank}: OK")

jax.distributed.shutdown()
