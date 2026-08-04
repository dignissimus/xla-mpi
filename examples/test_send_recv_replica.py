"""Cross-process stablehlo.send/stablehlo.recv.

Tests with (num_replicas=N, num_partitions=1) using a StableHLO module,
with the Client.compile_and_load() API.

The same program runs on both ranks: rank 0 is the source
and rank 1 is the target. Both ranks execute both the send and the recv op
(same code, no per-rank branching), but each op only has a real effect on
the rank matching its role. Rank 0's own recv zero-fills since it isn't a
target, rank 1's own send is a no-op since it isn't a source.
"""

import numpy as np
import jax
from jax._src.lib import xla_client as xc

jax.config.update("jax_platforms", "mpi")
jax.distributed.initialize(cluster_detection_method="mpi4py")

rank = jax.process_index()
size = jax.process_count()
assert size == 2, "test_send_recv_replica.py requires -np 2"

print(f"Rank {rank}: Devices = {jax.devices()}")

stablehlo_text = """
module @verify_send_recv attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 2 : i32} {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %token0 = "stablehlo.after_all"() : () -> !stablehlo.token
    %send_token = "stablehlo.send"(%arg0, %token0) {
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 1>,
      is_host_transfer = false,
      source_target_pairs = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<f32>, !stablehlo.token) -> !stablehlo.token
    %result:2 = "stablehlo.recv"(%token0) {
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 1>,
      is_host_transfer = false,
      source_target_pairs = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (!stablehlo.token) -> (tensor<f32>, !stablehlo.token)
    return %result#0 : tensor<f32>
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

local_data = jax.device_put(np.array(10.0 * rank + 5.0, dtype=np.float32), jax.devices()[rank])
result = executable.execute([local_data])
got = np.asarray(result[0])
expected = 5.0 if rank == 1 else 0.0  # rank 1 receives rank 0's value; rank 0's own recv zero-fills.
print(f"Rank {rank}: result = {got}; expected = {expected}")
assert got == expected, f"Rank {rank}: expected {expected}, got {got}"
print(f"Rank {rank}: OK")

jax.distributed.shutdown()
