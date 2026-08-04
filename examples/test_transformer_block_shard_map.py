"""Megatron-LM sharding scheme, using jax.experimental.shard_map and jax.lax.psum.

-np 2, num_heads = 2 (1 head/rank), d_model=4, d_ff=4 (2/rank), seq_len=3.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_platforms", "mpi")
jax.distributed.initialize(cluster_detection_method="mpi4py")

rank = jax.process_index()
size = jax.process_count()
assert size == 2, "test_transformer_block_shard_map.py requires -np 2"

print(f"Rank {rank}: Devices = {jax.devices()}")

d_model, num_heads, seq_len, d_ff = 4, 2, 3, 4
head_dim = d_model // num_heads

mesh = Mesh(np.array(jax.devices()).reshape(size), axis_names=("i",))


def sharded(local_np, global_shape, pspec):
    sharding = NamedSharding(mesh, pspec)
    local_dev = jax.device_put(jnp.asarray(local_np, dtype=jnp.float32), jax.local_devices()[0])
    return jax.make_array_from_single_device_arrays(global_shape, sharding, [local_dev])


x_np = np.array([[0.3047, -1.04, 0.7505, 0.9406],
                  [-1.951, -1.3022, 0.1278, -0.3162],
                  [-0.0168, -0.853, 0.8794, 0.7778]])

Wq_np = {0: np.array([[0.066, 1.1272], [0.4675, -0.8593], [0.3688, -0.9589], [0.8785, -0.0499]]),
         1: np.array([[-0.1849, -0.6809], [1.2225, -0.1545], [-0.4283, -0.3521], [0.5323, 0.3654]])}
Wk_np = {0: np.array([[0.4127, 0.4308], [2.1416, -0.4064], [-0.5122, -0.8138], [0.616, 1.129]]),
         1: np.array([[-0.1139, -0.8402], [-0.8245, 0.6506], [0.7433, 0.5432], [-0.6655, 0.2322]])}
Wv_np = {0: np.array([[0.1167, 0.2187], [0.8714, 0.2236], [0.6789, 0.0676], [0.2891, 0.6313]]),
         1: np.array([[-1.4572, -0.3197], [-0.4704, -0.6389], [-0.2751, 1.4949], [-0.8658, 0.9683]])}
Wo_np = {0: np.array([[-1.6829, -0.3349, 0.1628, 0.5862], [0.7112, 0.7933, -0.3487, -0.4624]]),
         1: np.array([[0.858, -0.1913, -1.2757, -1.1333], [-0.9195, 0.4972, 0.1424, 0.6905]])}
W1_np = {0: np.array([[-0.4273, 0.1585], [0.6256, -0.3093], [0.4568, -0.6619], [-0.3631, -0.3817]]),
         1: np.array([[-1.1958, 0.487], [-0.4694, 0.0125], [0.4807, 0.4465], [0.6654, -0.0985]])}
W2_np = {0: np.array([[-0.4233, -0.0797, -1.6873, -1.4471], [-1.3227, -0.9972, 0.3998, -0.9055]]),
         1: np.array([[-0.3782, 1.2992, -0.3563, 0.7375], [-0.9336, -0.2054, -0.95, -0.339]])}

x_s = sharded(x_np, (seq_len, d_model), P())
Wq_s = sharded(Wq_np[rank], (d_model, d_model), P(None, "i"))
Wk_s = sharded(Wk_np[rank], (d_model, d_model), P(None, "i"))
Wv_s = sharded(Wv_np[rank], (d_model, d_model), P(None, "i"))
Wo_s = sharded(Wo_np[rank], (d_model, d_model), P("i", None))
W1_s = sharded(W1_np[rank], (d_model, d_ff), P(None, "i"))
W2_s = sharded(W2_np[rank], (d_ff, d_model), P("i", None))


def gelu(z):
    return 0.5 * z * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (z + 0.044715 * z**3)))


def transformer_block(x, Wq, Wk, Wv, Wo, W1, W2):
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv
    scores = (q @ k.T) / jnp.sqrt(jnp.array(head_dim, dtype=jnp.float32))
    attn = jax.nn.softmax(scores, axis=-1) @ v
    partial_attn_out = attn @ Wo
    attn_output = jax.lax.psum(partial_attn_out, axis_name="i")
    x1 = x + attn_output

    h = gelu(x1 @ W1)
    partial_mlp_out = h @ W2
    mlp_output = jax.lax.psum(partial_mlp_out, axis_name="i")
    x2 = x1 + mlp_output
    return x2


sharded_block = shard_map(
    transformer_block, mesh=mesh,
    in_specs=(P(), P(None, "i"), P(None, "i"), P(None, "i"), P("i", None), P(None, "i"), P("i", None)),
    out_specs=P())

result = sharded_block(x_s, Wq_s, Wk_s, Wv_s, Wo_s, W1_s, W2_s)
got = np.asarray(result)

expected = np.array([
    [-2.1946813511, 4.4218970166, -0.5782163436, 4.2245646288],
    [-2.7013215234, 2.0667300239, -1.263676682, 1.2447132126],
    [-2.2986851428, 4.3302579203, -0.8146455577, 3.5649536254],
])

print(f"Rank {rank}: got=\n{got}\nexpected=\n{expected}")
assert np.allclose(got, expected, atol=1e-4), f"Rank {rank}: mismatch, diff={np.abs(got - expected).max()}"
print(f"Rank {rank}: OK")

jax.distributed.shutdown()
