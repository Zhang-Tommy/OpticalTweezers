import numpy as np
from batch_phase_generation import compute_all_pixels, batch_compute
import matplotlib.pyplot as plt
import h5py
import jax
import jax.numpy as jnp
import time

N = 100000 # total samples
B = 200 # batch size (10 x 4 x 4)
num_spots = 1

f = h5py.File("./data/train_data_1_spot.hdf5", "w")
inputs = f.create_dataset("inputs", (N, num_spots, 4, 4))
outputs = f.create_dataset("outputs", (N, 512, 512))

ranges = np.array([
    [0, 120],   # x
    [-90, 0],     # y
    [-8, 8],     # z
    [0, 12],     # l
    [0.1, 1],    # I
    [0, 2*np.pi],# p
    [0, 0],      # null
    [0, 0],      # null
    [0, 0],      # nax (fixed)
    [0, 0],      # nay (fixed)
    [1, 1],      # nar (fixed)
    [0, 0],      # null
    [0, 10],     # lx
    [0, 10],     # ly
    [0, 0],      # lz (fixed)
    [0, 0],      # lp (fixed)
])

@jax.jit
def scan_func(carry, x):
    new_sample = jax.random.uniform(jax.random.PRNGKey(x), (num_spots, 4, 4), minval=l, maxval=h, dtype=jnp.float32)
    carry = carry.at[x].set(new_sample)
    return carry, None

xs = jnp.arange(B)

low, high = ranges[:, 0], ranges[:, 1]
l = low.reshape(1, 4, 4)
h = high.reshape(1, 4, 4)

# Expected output: (B, 10, 4, 4)
for i in range(int(N / B)):
    st = time.time()

    batch = jnp.zeros((B, num_spots, 4, 4), dtype=jnp.float32)
    batch, _ = jax.lax.scan(scan_func, batch, xs)

    print(batch.shape)
    result = batch_compute(batch)

    inputs[i*B:(i+1)*B, :, :] = batch
    outputs[i*B:(i+1)*B, :, :] = result[0]

    et = time.time()

    #print(et - st)


#print(result[0].shape)
#
# for i in range(100):
#     samples = np.random.uniform(low, high, size=(10, 16)).reshape(10, 4, 4).astype(np.float32) # (10, 4, 4) single sample input
#     result = compute_all_pixels(samples)
#
#     print(result[0].shape)
#
#     inputs[i*batch_size:(i+1)*batch_size, :, :] = samples
#     outputs[i*batch_size:(i+1)*batch_size, :, :] = result[0]

# plt.figure(figsize=(5, 5))
# plt.title("Phase Pattern")
# plt.imshow(result[0][2], cmap='binary')
# plt.colorbar(label="Phase (radians)")

# plt.tight_layout()
# plt.show()