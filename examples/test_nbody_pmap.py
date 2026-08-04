"""N-body gravitational simulation via jax.pmap + jax.lax.all_gather.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_platforms", "mpi")
jax.distributed.initialize(cluster_detection_method="mpi4py")

rank = jax.process_index()
size = jax.process_count()
assert size == 2, "test_nbody_pmap.py requires -np 2"

print(f"Rank {rank}: Devices = {jax.devices()}")

N = 4
PER_RANK = N // size
G = 1.0
EPS2 = 0.01
DT = 0.01

all_pos = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32)
all_mass = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
all_vel = jnp.zeros((N, 2), dtype=jnp.float32)

local_pos = all_pos[rank * PER_RANK:(rank + 1) * PER_RANK]
local_mass = all_mass[rank * PER_RANK:(rank + 1) * PER_RANK]
local_vel = all_vel[rank * PER_RANK:(rank + 1) * PER_RANK]


def nbody_step(local_pos, local_mass, local_vel):
    global_pos = jax.lax.all_gather(local_pos, axis_name="i").reshape(N, 2)
    global_mass = jax.lax.all_gather(local_mass, axis_name="i").reshape(N)
    diff = global_pos[None, :, :] - local_pos[:, None, :]  # (PER_RANK, N, 2)
    r2 = jnp.sum(diff**2, axis=-1) + EPS2  # (PER_RANK, N)
    inv_r3 = r2**-1.5
    acc = jnp.sum(G * global_mass[None, :, None] * diff * inv_r3[:, :, None], axis=1)
    vel_new = local_vel + acc * DT
    pos_new = local_pos + vel_new * DT
    return pos_new


nbody_step_p = jax.pmap(nbody_step, axis_name="i")

result = nbody_step_p(local_pos[None], local_mass[None], local_vel[None])
got = np.asarray(result[0])

expected_full = np.array([
    [3.3740435411e-04, 4.3592288779e-04],
    [9.9979620600e-01, 4.9934959979e-04],
    [4.6425777811e-04, 9.9983129782e-01],
    [9.9966935258e-01, 9.9976787111e-01],
])
expected = expected_full[rank * PER_RANK:(rank + 1) * PER_RANK]

print(f"Rank {rank}: got=\n{got}\nexpected=\n{expected}")
assert np.allclose(got, expected, atol=1e-5), f"Rank {rank}: mismatch, diff={np.abs(got - expected).max()}"
print(f"Rank {rank}: OK")

jax.distributed.shutdown()
