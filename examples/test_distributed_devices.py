import os
import jax
import jax.numpy as jnp

jax.config.update("jax_platforms", "mpi")
jax.distributed.initialize(cluster_detection_method="mpi4py")

rank = jax.process_index()
size = jax.process_count()
print("hm", rank, size)

print(f"Rank {rank}: Devices = {jax.devices()}; Local devices: {jax.local_devices()}")
# TODO: Implement
# jax.distributed.shutdown()

# TODO: Error somewhere on exit
