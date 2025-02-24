import numpy as np
import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt


# Constants
pi = np.pi
white = np.array([1, 1, 1, 1])  # RGBA white color
n = 10  # Number of spots
totalA = 1.0

centre = np.array([0.5, 0.5])  # Center of the hologram
size = np.array([10200, 10200])  # Size of the hologram in microns
f = 4500.0  # Focal length in microns
k = 5.905249  # Wavevector in 1/microns
blazing = np.array([0.47451, 0.48716, 0.495889, 0.508539, 0.517647, 0.529412, 0.538646, 0.54902,
                    0.560025, 0.572549, 0.581404, 0.594054, 0.606705, 0.615686, 0.628083, 0.640734,
                    0.657306, 0.67451, 0.690449, 0.709804, 0.727514, 0.744086, 0.760784, 0.780392,
                    0.796078, 0.814295, 0.834788, 0.85136, 0.875775, 0.918027, 0.979507, 1.0])

zernikeCoefficients = np.array([[5, -17, 24.01, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

resolution = 512

def calculate_zernike(x, y):
    r2 = x**2 + y**2
    zerna = jnp.array([2.0 * x * y, 2.0 * r2 - 1.0, x**2 - y**2, 3.0 * x**2 * y - y**3])
    zernb = jnp.array([(3.0 * r2 - 2.0) * y, (3.0 * r2 - 2.0) * x, x**3 - 3.0 * x * y**2, 4.0 * x * y * (x**2 - y**2)])
    zernc = jnp.array([(4.0 * r2 - 3.0) * zerna[0], 6.0 * r2**2 - 6.0 * r2 + 1, (4.0 * r2 - 3.0) * zerna[2], x**4 - 6.0 * x**2 * y**2 + y**4])
    return zerna, zernb, zernc

@jax.jit
def compute_pixel(px, py, spots):
    xy = ((jnp.array([px, py]) / resolution) - centre) * size
    phi = jnp.arctan2(xy[1], xy[0])
    pos = jnp.array([xy[0] / f, xy[1] / f, 1.0 - jnp.dot(xy, xy) / (2.0 * f ** 2), phi / k])
    x, y = (xy / size) * 2.0
    zerna, zernb, zernc = calculate_zernike(x, y)

    xs = jnp.arange(0, n, 1).astype(jnp.int32) # [0 1 2 3 4 5 6 7 8 9] spots will be (10, 4, 4)

    def spot_f(i):
        #return jax.lax.dynamic_slice(spots, (i, 0, 0), (4, 4))
        return jax.lax.dynamic_index_in_dim(spots, i, axis=0, keepdims=False)

    def scan_func_ri(carry, x): # x iterates through spots
        totalr, totali = carry
        spot_data = spot_f(x)  # Get single spot data using dynamic slice
        #jax.debug.print("spot_data: {spot_data} x: {x}", spot_data=spot_data, x=x)

        amplitude = spot_data[1, 0]
        #jax.debug.print("amplitude: {amplitude}", amplitude=amplitude)
        phase = k * jnp.dot(spot_data[0, :], pos) + spot_data[1, 1]
        na = spot_data[2, :3]

        # Perform NA restriction
        # if jnp.dot(na[:2] - xy / size, na[:2] - xy / size) > na[2] ** 2:
        #    amplitude = 0.0

        # Line trap logic
        line = spot_data[3, :]
        length = jnp.sqrt(jnp.dot(line[:3], line[:3]))
        # if length > 0.0:
        #    sx = k * jnp.dot(jnp.append(pos[:3], length), line)
        #    if sx != 0.0:
        #        amplitude *= jnp.sin(sx) / sx

        # Convert amplitude + phase to real + imaginary
        totalr += amplitude * jnp.sin(phase)
        totali += amplitude * jnp.cos(phase)

        carry = (totalr, totali)
        return carry, (totalr, totali)

    final, result = jax.lax.scan(scan_func_ri, (0, 0), xs)
    #jax.debug.print("result_dim: {result_dim}", result_dim=result[0].shape)
    #jax.debug.print("final: {final}", final=final[0].shape)
    totalr = result[0][9]  # result contains each cumulative sum as it steps through the for loop over number of spots
    totali = result[1][9]

    amplitude = jnp.sqrt(totalr ** 2 + totali ** 2)
    phase = jnp.arctan2(totalr, totali)
    # if amplitude == 0.0:
    #    phase = 0.0
    # if totalA > 0.0:
    #    phase *= jnp.clip(amplitude / totalA, 0.0, 1.0)
    phase += jnp.dot(zernikeCoefficients[0], zerna) + jnp.dot(zernikeCoefficients[1], zernb) + jnp.dot(
        zernikeCoefficients[2], zernc)
    phase = (phase + pi) % (2 * pi) - pi

    # Convert phase to blazing table
    phint = (jnp.floor((phase / (2 * pi) + 0.5) * (len(blazing) - 1))).astype(int)
    alpha = (phase / (2 * pi) + 0.5) * (len(blazing) - 1) - phint
    # pixel_value = white * blazing[phint] * (1.0 - alpha) + blazing[phint + 1] * alpha

    # print(phase)
    # Store values
    # phase_pattern[px, py] = phase
    # amplitude_pattern[px, py] = amplitude

    # print(result[0])
    return phase, amplitude

@jax.jit
def compute_row(row, spots):
    cols = jnp.arange(resolution)
    return jax.vmap(lambda col: compute_pixel(row, col, spots))(cols)

@jax.jit
def compute_all_pixels(spots):
    rows = jnp.arange(resolution)
    return jax.vmap(lambda row: compute_row(row, spots))(rows)

@jax.jit
def batch_compute(spots_batch):
    """ (batch_size, n_spots, 4, 4) """
    return jax.vmap(compute_all_pixels)(spots_batch)


if __name__ == "__main__":
    spots_batch = np.random.rand(10, 4, 4)  # Batch of 5 sets of spot configurations
    # spots = np.array([[12, -50.078036, 0.000000, 5.000000],
    #                       [1.000000, 0.000000, 0.000000, 0.000000],
    #                       [0.000000, 0.000000, 1.000000, 0.000000],
    #                       [0.000000, 0.000000, 0.000000, 0.000000]])

    #spots_batch = np.tile(spots, (10, 1, 1))

    """
    x [-120, 0]
    y [0, 90]
    z [-8, 8]
    I [0.1, 1]
    p [0, 2pi]
    nax nay [0,0]
    nar [1]
    lx ly [0, 10]
    lz lp [0]
    
    """

    """
    x y z l
    I p - -
    nax nay nar -
    lx ly lz lp
    """
    spots = np.array([

        [[6, -45.210987, 0.000000, 5.800000],
         [1.100000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 1.150000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],

        [[4, -50.078036, 0.000000, 5.000000],
         [0.600000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 1.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],

        [[1, -35.987654, 0.000000, 7.000000],
         [0.400000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 1.100000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],

        [[2, -41.123456, 0.000000, 6.500000],
         [0.400000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 1.200000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],

        [[1, -55.432198, 0.000000, 4.500000],
         [1.050000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.900000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],

        [[90, -30.876543, 0.000000, 8.000000],
         [0.950000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 1.300000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],



        [[4, -12.567890, 0.000000, 6.700000],
         [0.870000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 1.250000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],

        [[90, -32.098765, 0.000000, 7.500000],
         [0.920000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 1.050000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],

        [[90, -53.876543, 0.000000, 4.200000],
         [1.030000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.950000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],

        [[90, -42.654321, 0.000000, 6.200000],
         [0.880000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 1.180000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]]
    ])
    print(spots.shape)
    #spots_batch[0,:,:] = spots
    st = time.time()
    result = compute_all_pixels(spots)
    et = time.time()

    print("Time taken:", et - st)

    print(result[1].shape)
    plt.figure(figsize=(5, 5))
    plt.title("Phase Pattern")
    plt.imshow(result[0], cmap='binary')
    plt.colorbar(label="Phase (radians)")

    plt.tight_layout()
    plt.show()

