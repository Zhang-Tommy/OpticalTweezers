import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
from phase_generator import batch_calculate_masks, calculate_phase_mask
import h5py
from scipy.special import j1
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import phase_generator_2

def gen_airy(n, radius=16):
    """
    Generate a 2D Airy pattern of size n x n.
    radius: Controls the size of the central lobe.
    """
    x = np.linspace(-n//2, n//2, n)
    y = np.linspace(-n//2, n//2, n)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2) / radius  # Normalize by radius
    airy = (2 * j1(r) / r) ** 2  # Airy disk formula
    airy[r == 0] = 1.0  # Handle division by zero at the center
    return airy

def add_airy_to_mask(feature_mask, airy, x_center, y_center, mask_size):
    """
    Adds a 2D Airy pattern to feature_mask at (x_center, y_center).
    """
    n = airy.shape[0]
    half_n = n // 2

    # Define the region in feature_mask
    x_min = max(x_center - half_n, 0)
    x_max = min(x_center + half_n + 1, mask_size)
    y_min = max(y_center - half_n, 0)
    y_max = min(y_center + half_n + 1, mask_size)

    # Define the corresponding region in the Airy pattern
    a_x_min = max(0, half_n - x_center)
    a_x_max = a_x_min + (x_max - x_min)
    a_y_min = max(0, half_n - y_center)
    a_y_max = a_y_min + (y_max - y_min)

    # Add the Airy pattern to the feature mask
    feature_mask[x_min:x_max, y_min:y_max] += airy[a_x_min:a_x_max, a_y_min:a_y_max]

def gen_gaussian(n, sigma=16):
    """Generate a 2D Gaussian kernel of size n x n."""
    x = np.linspace(-n//2, n//2, n)
    y = np.linspace(-n//2, n//2, n)
    X, Y = np.meshgrid(x, y)
    gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return gaussian

def add_rectangle_to_mask(feature_mask, x_center, y_center, rect_width, rect_height, mask_size):
    """
    Adds a rectangle of intensity 1 to feature_mask at (x_center, y_center).

    Parameters:
        feature_mask (np.ndarray): The 2D array to which the rectangle will be added.
        x_center (int): The x-coordinate of the rectangle's center.
        y_center (int): The y-coordinate of the rectangle's center.
        rect_width (int): The width of the rectangle.
        rect_height (int): The height of the rectangle.
    """

    # Calculate the rectangle's boundaries
    half_width = rect_width // 2
    half_height = rect_height // 2

    x_min = max(x_center - half_width, 0)
    x_max = min(x_center + half_width + 1, mask_size)
    y_min = max(y_center - half_height, 0)
    y_max = min(y_center + half_height + 1, mask_size)

    # Add the rectangle to the feature mask
    feature_mask[x_min:x_max, y_min:y_max] = 1

    #return feature_mask

def add_gaussian_to_mask(feature_mask, gaussian, x_center, y_center, mask_size):
    """Adds a 2D Gaussian to feature_mask at (x_center, y_center)."""
    n = gaussian.shape[0]
    half_n = n // 2

    # Define the region in feature_mask
    x_min = max(x_center - half_n, 0)
    x_max = min(x_center + half_n + 1, mask_size)
    y_min = max(y_center - half_n, 0)
    y_max = min(y_center + half_n + 1, mask_size)

    # Define the corresponding region in the Gaussian
    g_x_min = max(0, half_n - x_center)
    g_x_max = g_x_min + (x_max - x_min)
    g_y_min = max(0, half_n - y_center)
    g_y_max = g_y_min + (y_max - y_min)

    # Add the Gaussian to the feature mask
    feature_mask[x_min:x_max, y_min:y_max] += gaussian[g_x_min:g_x_max, g_y_min:g_y_max]

    #feature_mask = feature_mask.at[x_min:x_max, y_min:y_max].add(gaussian[g_x_min:g_x_max, g_y_min:g_y_max])

def add_ring_intensity(x, y, x0, y0, r0, sigma, amplitude=1.0):
    """
    Ring-shaped intensity distribution with Gaussian-like dropoff.
    x, y: Input coordinates.
    x0, y0: Center of the ring.
    r0: Radius of the ring.
    sigma: Controls the width of the Gaussian-like dropoff.
    amplitude: Maximum intensity of the ring.
    """
    # Radial distance from the center
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    # Gaussian-like intensity distribution
    intensity = amplitude * np.exp(-((r - r0) ** 2) / (2 * sigma ** 2))

    return intensity


def logi(intensity):
    # Normalize by max value per batch
    intensity = intensity / jnp.max(intensity, axis=(1, 2), keepdims=True)

    # Crop the right-bottom quarter (256:512, 256:512) from each image
    intensity = intensity[:, 256:, 256:]

    # Apply square root transform for better visibility
    intensity = jnp.power(intensity, 1 / 2)

    # Rotate each image 270 degrees (equivalent to transposing + flipping vertically)
    intensity = jnp.rot90(intensity, 3, axes=(1, 2))

    # Flip each image both horizontally and vertically
    intensity = jnp.flip(intensity, axis=(1, 2))

    return intensity

#mask_size = 512
N = 10000
B = 500
#resolutions = [32, 64, 128, 256]
resolutions = [512]

### Generate data for slm resolution of [16, 32, 64, 128]
for mask_size in resolutions:
    @jax.jit
    def scan_func(carry, inputs):
        key, x = inputs  # Extract key and loop index
        key, subkey = jax.random.split(key)  # Split key for randomness

        new_sample = jax.random.uniform(subkey, (num_spots, 4, 4), minval=l, maxval=h, dtype=jnp.float32)

        # Donut choices and probabilities
        key, subkey = jax.random.split(key)
        donut_choice = jnp.array([0, int((mask_size / 512) * 15)])
        donut_probs = jnp.array([0.8, 0.2])  # 70% chance of 0, 30% chance of 7
        d_idx = jax.random.choice(subkey, len(donut_choice), p=donut_probs, shape=(num_spots,))
        new_sample = new_sample.at[:, 0, 3].set(donut_choice[d_idx])

        # Line choices and probabilities
        key, subkey = jax.random.split(key)
        line_choice = jnp.array([0, 3])
        line_probs = jnp.array([0.8, 0.2])  # 80% chance of 0, 20% chance of 3
        l_idx = jax.random.choice(subkey, len(line_choice), p=line_probs, shape=(num_spots,))
        new_sample = new_sample.at[:, 3, 0].set(line_choice[l_idx])

        carry = carry.at[x].set(new_sample)
        return carry, key  # Return updated carry and key to prevent reuse

    print(f"Generating data for: {mask_size} x {mask_size} resolution")
    sigma = int((mask_size / 512) * 8)
    gaussian = gen_gaussian(255, sigma)
    airy = gen_airy(255, sigma)
    pos_scale = mask_size / 120
    line_trap_width = int((mask_size / 512) * 16)
    line_trap_height = int((mask_size / 512) * 70)
    ring_trap_dia = int((mask_size / 512) * 30)
    ring_trap_sigma = (mask_size / 512) * 13
    num_spots = 10

    f = h5py.File(f"./train_data/unet_{N}_{mask_size}.hdf5", "w")
    inputs = f.create_dataset("inputs", (N, mask_size, mask_size))
    outputs = f.create_dataset("outputs", (N, mask_size, mask_size))

    ranges = np.array([
        [0, 120],   # x
        [-90, 0],     # y
        [0, 0],     # z
        [0, 0],     # l
        [1, 1],     # I
        [0, 0],     # p
        [0, 0],      # null
        [0, 0],      # null
        [0, 0],      # nax (fixed)
        [0, 0],      # nay (fixed)
        [1, 1],      # nar (fixed)
        [0, 0],      # null
        [0, 0],     # lx
        [0, 0],     # ly
        [0, 0],      # lz (fixed)
        [0, 0],      # lp (fixed)
    ])

    xs = jnp.arange(B)

    low, high = ranges[:, 0], ranges[:, 1]
    l = low.reshape(1, 4, 4)
    h = high.reshape(1, 4, 4)

    idx = 0
    # Expected output: (B, 10, 4, 4)
    for j in range(1, num_spots + 1):
        # for every batch of input/outputs
        input_batch = np.zeros((B, mask_size, mask_size))
        for i in range(int((N // B) // num_spots)):
            batch = jnp.zeros((B, num_spots, 4, 4), dtype=jnp.float32)
            # batch always has max number of spots

            key = jax.random.PRNGKey(i*j)  # Initialize a key

            xs = jnp.arange(B)
            keys = jax.random.split(key, B)  # Generate unique keys for each iteration

            batch, _ = jax.lax.scan(scan_func, batch, (keys, xs))  # Pass keys along with xs
            mask = jnp.arange(num_spots) < j  # Creates a mask where indices < j are True
            mask = mask.astype(jnp.float32)  # Convert to float to allow multiplication
            mask = mask[None, :, None, None]  # Reshape for broadcasting over batch

            batch = batch * mask  # Zero out unwanted indices

            # # each batch has 200 spot_parameters
            # k = 0
            # # each loop iteration generates single feature_mask
            # for spot_params in batch:
            #     feature_mask = np.zeros((mask_size, mask_size))
            #     for trap in spot_params:
            #         x_center = int(trap[0, 0] * pos_scale)
            #         y_center = -int(trap[0, 1] * pos_scale)
            #         if trap[0, 0] == 0 and trap[0, 1] == 0:
            #              pass
            #         elif trap[3,0] != 0:
            #             add_rectangle_to_mask(feature_mask, x_center, y_center, line_trap_width, line_trap_height, mask_size)
            #         elif trap[0,3] != 0:
            #             x = jnp.linspace(0, mask_size, mask_size)
            #             y = jnp.linspace(0, mask_size, mask_size)
            #             X, Y = jnp.meshgrid(x, y)
            #             #print(x_center, y_center)
            #
            #             feature_mask += add_ring_intensity(X, Y, x_center, y_center, ring_trap_dia, ring_trap_sigma, 1)
            #         else:
            #             #add_gaussian_to_mask(feature_mask, gaussian, y_center, x_center, mask_size)
            #             add_airy_to_mask(feature_mask, airy, y_center, x_center, mask_size)
            #
            #     input_batch[k, :, :] = feature_mask / (np.max(feature_mask))
            #

            #    k += 1

            phase_masks_corrected = phase_generator_2.batch_calculate_masks(batch[:, 0:j, :, :], j, mask_size)[0]
            phase_masks_reference = batch_calculate_masks(batch[:, 0:j, :, :], j, mask_size)[0]

            far_field = fftshift(fft2(jnp.exp(1j * phase_masks_reference)))
            intensity = jnp.abs(far_field) ** 2

            intensity = logi(intensity)

            intensity_upscaled = jax.image.resize(
                intensity,
                shape=(intensity.shape[0], mask_size, mask_size),
                method="nearest"  # or "nearest", "lanczos"
            )
            #print(np.max(intensity_upscaled[0,:,:]))
            #print(intensity.shape)

            plt.imshow(intensity[0,:,:], cmap='hot')
            plt.colorbar()
            plt.show()

            inputs[idx * B:(idx + 1) * B, :, :] = intensity_upscaled
            outputs[idx * B:(idx + 1) * B, :, :] = phase_masks_corrected / np.pi
            print(f"Batch: {idx+1} finished. {(idx + 1)*B} / {N}")
            idx += 1

