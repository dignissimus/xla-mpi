import os
import jax
import jax.numpy as jnp

jax.config.update("jax_platforms", "mpi")
jax.distributed.initialize(cluster_detection_method="mpi4py")

rank = jax.process_index()
size = jax.process_count()

print(f"Rank {rank}: Devices = {jax.devices()}; Local devices: {jax.local_devices()}")
local_data = jnp.array([rank], dtype=jnp.float32)

@jax.pmap(axis_name='i')
def test_allreduce(x):
    return jax.lax.psum(x, axis_name='i')

try:
    result = test_allreduce(local_data)
    expected_sum = sum(range(size))
    
    print(f"Rank {rank}: psum result = {result[0]}; Expected = {expected_sum}")
    
    if result[0] == expected_sum:
        print(f"Rank {rank}: correct result")
    else:
        print(f"Rank {rank}: wrong result")
        
except Exception as e:
    print(f"Rank {rank}: Error = {e}")

jax.distributed.shutdown()
