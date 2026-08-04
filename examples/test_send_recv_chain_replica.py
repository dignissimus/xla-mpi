"""stablehlo.send/recv with more than one pair.

Requires -np 4.
Chain: 0->1->2, rank 3 is neither a source nor a target of
any pair.
"""

import numpy as np
import jax
from jax._src.lib import xla_client as xc

jax.config.update("jax_platforms", "mpi")
jax.distributed.initialize(cluster_detection_method="mpi4py")

rank = jax.process_index()
size = jax.process_count()
assert size == 4, "test_send_recv_chain_replica.py requires -np 4"

print(f"Rank {rank}: Devices = {jax.devices()}")

stablehlo_text = """
module @verify_send_recv_chain attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 4 : i32} {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %token0 = "stablehlo.after_all"() : () -> !stablehlo.token
    %send_token = "stablehlo.send"(%arg0, %token0) {
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 1>,
      is_host_transfer = false,
      source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>
    } : (tensor<f32>, !stablehlo.token) -> !stablehlo.token
    %result:2 = "stablehlo.recv"(%token0) {
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 1>,
      is_host_transfer = false,
      source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>
    } : (!stablehlo.token) -> (tensor<f32>, !stablehlo.token)
    return %result#0 : tensor<f32>
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
# rank 0: source only -> its own recv zero-fills.
# rank 1: recv target of (0,1) -> gets rank 0's value (100.0).
# rank 2: recv target of (1,2) -> gets rank 1's value (101.0).
# rank 3: not involved in any pair -> zero-fills.
expected = {0: 0.0, 1: 100.0, 2: 101.0, 3: 0.0}[rank]
print(f"Rank {rank}: result = {got}; expected = {expected}")
assert got == expected, f"Rank {rank}: expected {expected}, got {got}"
print(f"Rank {rank}: OK")

jax.distributed.shutdown()
