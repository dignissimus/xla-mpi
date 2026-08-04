"""data-parallel gradient averaging for a small 2-layer MLP.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_platforms", "mpi")
jax.distributed.initialize(cluster_detection_method="mpi4py")

rank = jax.process_index()
size = jax.process_count()
assert size == 2, "test_data_parallel_grad_replica.py requires -np 2"

print(f"Rank {rank}: Devices = {jax.devices()}")

W1 = jnp.array([[1.2000e-03, 2.9870e-01, -2.7410e-01, -8.9060e-01],
                [-4.5470e-01, -9.9160e-01, 6.0100e-02, 1.3402e+00],
                [-4.9220e-01, -6.2050e-01, 4.8980e-01, 3.5690e-01]])
b1 = jnp.array([0.1054, -0.9305, -0.0293, 0.6953])
W2 = jnp.array([[-1.3442, -0.4576], [-1.9012, -1.2895], [-1.8417, -0.2351], [-1.2674, 0.2713]])
b2 = jnp.array([0.1568, -0.1869])

x_local_np = {0: np.array([[-2.5168, -0.5387, -0.0485], [0.1133, -1.5301, -0.4778]]),
              1: np.array([[-0.9785, -0.8088, 1.0609], [-0.8075, -0.0325, 0.8844]])}
y_local_np = {0: np.array([[-0.5836, -0.1117], [0.1105, 0.0638]]),
              1: np.array([[-1.2251, 0.0761], [1.3588, -1.5471]])}

x_local = jnp.asarray(x_local_np[rank])
y_local = jnp.asarray(y_local_np[rank])


def loss_fn(params, x, y):
    W1, b1, W2, b2 = params
    z1 = x @ W1 + b1
    a1 = jax.nn.relu(z1)
    pred = a1 @ W2 + b2
    return jnp.mean((pred - y)**2)


def avg_grad_step(x, y):
    params = (W1, b1, W2, b2)
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree.map(lambda g: jax.lax.psum(g, axis_name="i") / size, grads)


avg_grad_step_p = jax.pmap(avg_grad_step, axis_name="i")

avg_dW1, avg_db1, avg_dW2, avg_db2 = avg_grad_step_p(x_local[None], y_local[None])

got_dW1 = np.asarray(avg_dW1[0])
got_db1 = np.asarray(avg_db1[0])
got_dW2 = np.asarray(avg_dW2[0])
got_db2 = np.asarray(avg_db2[0])

exp_dW1 = np.array([[-2.890265511, 0.2360961749, -6.2652931337, -4.4938641303],
                     [-2.5720758398, -3.1884444594, -1.3525248567, -0.9316560084],
                     [-0.6589184582, -0.9956465347, 2.1685040344, 1.6286924777]])
exp_db1 = np.array([2.4617202329, 2.08381443, 4.1192349269, 3.0170637209])
exp_dW2 = np.array([[-1.1401134965, -0.4748666551], [-0.7085273829, -0.4373327186],
                     [-1.4323339965, 0.2571659913], [-4.1200985862, 0.7772274762]])
exp_db2 = np.array([-3.0629954938, -0.0558701791])

print(f"Rank {rank}: got_dW1=\n{got_dW1}\nexp_dW1=\n{exp_dW1}\nratio=\n{got_dW1 / exp_dW1}")
print(f"Rank {rank}: dW1 max diff = {np.abs(got_dW1 - exp_dW1).max()}")
assert np.allclose(got_dW1, exp_dW1, atol=1e-4), f"Rank {rank}: dW1 mismatch"
assert np.allclose(got_db1, exp_db1, atol=1e-4), f"Rank {rank}: db1 mismatch"
assert np.allclose(got_dW2, exp_dW2, atol=1e-4), f"Rank {rank}: dW2 mismatch"
assert np.allclose(got_db2, exp_db2, atol=1e-4), f"Rank {rank}: db2 mismatch"
print(f"Rank {rank}: OK")

jax.distributed.shutdown()
