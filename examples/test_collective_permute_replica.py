"""Cross-process collective_permute (Isend/Irecv).

Tests the async Start/Done custom-call with (num_replicas=N,
num_partitions=1) using a StableHLO module, with the
Client.compile_and_load() API.
Runs stablehlo.collective_permute with a cyclic shift: rank i sends its
own scalar to rank (i+1) % size, so rank i's result should be the scalar
rank (i-1+size) % size sent (i.e. its own rank value shifted by one).
"""

import numpy as np
import jax
from jax._src.lib import xla_client as xc

jax.config.update("jax_platforms", "mpi")
jax.distributed.initialize(cluster_detection_method="mpi4py")

rank = jax.process_index()
size = jax.process_count()

print(f"Rank {rank}: Devices = {jax.devices()}")

pairs = ", ".join(f"[{i}, {(i + 1) % size}]" for i in range(size))
stablehlo_text = f"""
module @verify_collective_permute attributes {{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = {size} : i32}} {{
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {{
    %0 = "stablehlo.collective_permute"(%arg0) {{source_target_pairs = dense<[{pairs}]> : tensor<{size}x2xi64>}}
      : (tensor<f32>) -> tensor<f32>
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
expected = float((rank - 1 + size) % size)
print(f"Rank {rank}: result = {got}; expected = {expected}")
assert got == expected, f"Rank {rank}: expected {expected}, got {got}"
print(f"Rank {rank}: OK")

jax.distributed.shutdown()
