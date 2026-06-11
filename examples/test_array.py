import os
import jax
import jax.numpy as jnp

jax.config.update("jax_platforms", "mpi")
jax.distributed.initialize(cluster_detection_method="mpi4py")

rank = jax.process_index()
size = jax.process_count()

print(f"Rank {rank}: Devices = {jax.devices()}; Local devices: {jax.local_devices()}")
local_data = jnp.array([rank], dtype=jnp.float32)

jax.distributed.shutdown()
