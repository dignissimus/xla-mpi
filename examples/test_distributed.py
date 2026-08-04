import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platforms", "mpi")
jax.distributed.initialize(cluster_detection_method="mpi4py")

rank = jax.process_index()
size = jax.process_count()

print(f"Rank {rank}: Devices = {jax.devices()}; Local devices: {jax.local_devices()}")


def test_allreduce(x):
    return jax.lax.psum(x, axis_name="i")


test_allreduce = jax.pmap(test_allreduce, axis_name="i")


local_data = jnp.array([float(rank)], dtype=jnp.float32)
result = test_allreduce(local_data)

expected_sum = float(sum(range(size)))
got = np.asarray(result)[0]
print(f"Rank {rank}: psum result = {got}; expected = {expected_sum}")
assert got == expected_sum, f"Rank {rank}: expected {expected_sum}, got {got}"
print(f"Rank {rank}: OK")

jax.distributed.shutdown()
