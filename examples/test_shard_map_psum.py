import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_platforms", "mpi")
jax.distributed.initialize(cluster_detection_method="mpi4py")

rank = jax.process_index()
size = jax.process_count()

print(f"Rank {rank}: Devices = {jax.devices()}; Local devices: {jax.local_devices()}")

mesh = Mesh(np.array(jax.devices()).reshape(size), axis_names=("i",))


def psum_fn(x):
    return jax.lax.psum(x, axis_name="i")


sharded_psum = shard_map(psum_fn, mesh=mesh, in_specs=P("i"), out_specs=P())

sharding = NamedSharding(mesh, P("i"))
local_shard = jax.device_put(jnp.array([float(rank)], dtype=jnp.float32), jax.local_devices()[0])
local_data = jax.make_array_from_single_device_arrays((size,), sharding, [local_shard])
result = sharded_psum(local_data)

expected_sum = float(sum(range(size)))
got = np.asarray(result)[0]
print(f"Rank {rank}: shard_map psum result = {got}; expected = {expected_sum}")
assert got == expected_sum, f"Rank {rank}: expected {expected_sum}, got {got}"
print(f"Rank {rank}: OK")

jax.distributed.shutdown()
