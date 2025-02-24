from timeit import timeit

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import timeit
# Constants
pi = np.pi
white = np.array([1, 1, 1, 1])  # RGBA white color

# Uniform variables (initialize as needed)
n = 10  # Number of spots
totalA = 1.0

spots = np.array([11.570961, -50.078036, 0.000000, 5.000000,
                  1.000000, 0.000000, 0.000000, 0.000000,
                  0.000000, 0.000000, 1.000000, 0.000000,
                  0.000000, 0.000000, 0.000000, 0.000000])

# Reshape into a 4x4 array
spots_4x4 = spots.reshape((4, 4))

# Append zeros to make it sized (200, 4)
spots = np.zeros((200, 4))
spots[:4] = spots_4x4

centre = np.array([0.5, 0.5])  # Centre of the hologram
size = np.array([10200, 10200])  # Size of the hologram in microns
f = 4500.0  # Focal length in microns
k = 5.905249  # Wavevector in 1/microns
blazing = np.array([0.47451, 0.48716, 0.495889, 0.508539, 0.517647, 0.529412, 0.538646, 0.54902,
                    0.560025, 0.572549, 0.581404, 0.594054, 0.606705, 0.615686, 0.628083, 0.640734,
                    0.657306, 0.67451, 0.690449, 0.709804, 0.727514, 0.744086, 0.760784, 0.780392,
                    0.796078, 0.814295, 0.834788, 0.85136, 0.875775, 0.918027, 0.979507, 1.0])

zernikeCoefficients = np.array([[5, -17, 24.01, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
zernx = np.zeros(3)
zerny = np.zeros(3)
zernz = np.zeros(3)

resolution = 512

# Helper function to get spot parameters
def spot(i, j):
    return spots[4 * i + j]

def spot_f(i):
    return spots[4*i:4*i + 4, :]

# Zernike polynomials calculation
def calculate_zernike(x, y):
    r2 = x**2 + y**2
    zerna = jnp.array([2.0 * x * y, 2.0 * r2 - 1.0, x**2 - y**2, 3.0 * x**2 * y - y**3])
    zernb = jnp.array([(3.0 * r2 - 2.0) * y, (3.0 * r2 - 2.0) * x, x**3 - 3.0 * x * y**2, 4.0 * x * y * (x**2 - y**2)])
    zernc = jnp.array([(4.0 * r2 - 3.0) * zerna[0], 6.0 * r2**2 - 6.0 * r2 + 1, (4.0 * r2 - 3.0) * zerna[2], x**4 - 6.0 * x**2 * y**2 + y**4])
    return zerna, zernb, zernc

@jax.jit
def compute_pixel(px, py):
    xy = ((jnp.array([px, py]) / resolution) - centre) * size
    phi = jnp.arctan2(xy[1], xy[0])
    pos = jnp.array([xy[0] / f, xy[1] / f, 1.0 - jnp.dot(xy, xy) / (2.0 * f ** 2), phi / k])
    x, y = xy / size * 2.0
    zerna, zernb, zernc = calculate_zernike(x, y)

    # make array x one row is 1 2 3 4 ... n second row is spot(1) spot(2) ... spot(n)

    xs = jnp.arange(0, n, 1).astype(jnp.int32)

    def spot_f(i):
        return jax.lax.dynamic_slice(spots, (4 * i, 0), (4, 4))  # Start at (4 * i, 0) and slice (4, 4)

    def scan_func_ri(carry, x):
        totalr, totali, spots = carry
        spot_data = spot_f(x)  # Get spot data using dynamic slice
        #jax.debug.print("{spot_data}", spot_data=spot_data)
        amplitude = spot_data[0, 1]
        phase = k * jnp.dot(spot_data[0, :], pos) + spot_data[1, 1]
        na = spot_data[2, :3]

        # Perform NA restriction
        #if jnp.dot(na[:2] - xy / size, na[:2] - xy / size) > na[2] ** 2:
        #    amplitude = 0.0

        # Line trap logic
        line = spot_data[3, :]
        length = jnp.sqrt(jnp.dot(line[:3], line[:3]))
        #if length > 0.0:
        #    sx = k * jnp.dot(jnp.append(pos[:3], length), line)
        #    if sx != 0.0:
        #        amplitude *= jnp.sin(sx) / sx

        # Convert amplitude + phase to real + imaginary
        totalr += amplitude * jnp.sin(phase)
        totali += amplitude * jnp.cos(phase)

        carry = (totalr, totali, spots)
        return carry, (totalr, totali)

    final, result = jax.lax.scan(scan_func_ri, (0, 0, spots), xs)
    totalr = result[0][0]
    totali = result[1][0]

    amplitude = jnp.sqrt(totalr ** 2 + totali ** 2)
    phase = jnp.arctan2(totalr, totali)
    #if amplitude == 0.0:
    #    phase = 0.0
    #if totalA > 0.0:
    #    phase *= jnp.clip(amplitude / totalA, 0.0, 1.0)
    phase += jnp.dot(zernikeCoefficients[0], zerna) + jnp.dot(zernikeCoefficients[1], zernb) + jnp.dot(
        zernikeCoefficients[2], zernc)
    phase = (phase + pi) % (2 * pi) - pi

    # Convert phase to blazing table
    phint = (jnp.floor((phase / (2 * pi) + 0.5) * (len(blazing) - 1))).astype(int)
    alpha = (phase / (2 * pi) + 0.5) * (len(blazing) - 1) - phint
    #pixel_value = white * blazing[phint] * (1.0 - alpha) + blazing[phint + 1] * alpha

    #print(phase)
    # Store values
    #phase_pattern[px, py] = phase
    #amplitude_pattern[px, py] = amplitude



    #print(result[0])
    return phase, amplitude

@jax.jit
def compute_row(row):
    cols = jnp.arange(512)  # All columns
    return jax.vmap(lambda col: compute_pixel(row, col))(cols)

# Vectorized computation of the entire grid
@jax.jit
def compute_all_pixels():
    rows = jnp.arange(512)  # All rows
    return jax.vmap(compute_row)(rows)

# Compute the phase and amplitude patterns for all pixels
import time
st = time.time()


result = compute_all_pixels()


end = time.time()

print(end-st)

plt.figure(figsize=(5, 5))
plt.title("Phase Pattern")
plt.imshow(result[0], cmap='binary')
plt.colorbar(label="Phase (radians)")

plt.tight_layout()
plt.show()






