import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platforms", "mpi")
jax.distributed.initialize(cluster_detection_method="mpi4py")

rank = jax.process_index()


@jax.jit
def f(x, y):
    return x * 2.0 + y - 1.0


x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
y = jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32)
result = f(x, y)
expected = [11.0, 23.0, 35.0]

print(f"Rank {rank}: result = {result}")
got = np.asarray(result)
assert (got == np.array(expected, dtype=np.float32)).all(), f"Expected {expected}, got {got}"
print(f"Rank {rank}: OK")

jax.distributed.shutdown()
