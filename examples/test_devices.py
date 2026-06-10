import os
import jax

jax.config.update("jax_platforms", "mpi")
try:
    devices = jax.devices()
    print(f"Devices: {devices}")
except Exception as e:
    print(f"Error: {e}")
