# import numpy as np
# import jax
# import jax.numpy as jnp
# import time
# import matplotlib.pyplot as plt
# import functools
#
# slm_resolution = 64 # pixels resolution of SLM (square)
# center = np.array([0.5, 0.5]) # center of hologram
# hologram_size = np.array([10200, 10200]) # size of hologram (microns)
# f = 4500.0 # focal length (microns)
# k = 5.905249  # wavevector in 1/microns
# blazing = np.array([0.47451, 0.48716, 0.495889, 0.508539, 0.517647, 0.529412, 0.538646, 0.54902,
#                     0.560025, 0.572549, 0.581404, 0.594054, 0.606705, 0.615686, 0.628083, 0.640734,
#                     0.657306, 0.67451, 0.690449, 0.709804, 0.727514, 0.744086, 0.760784, 0.780392,
#                     0.796078, 0.814295, 0.834788, 0.85136, 0.875775, 0.918027, 0.979507, 1.0])
# zernikeCoefficients = np.array([[5, -17, 24.01, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
#
# def calculate_zernike(x, y):
#     r2 = x ** 2 + y ** 2
#     zerna = jnp.array([2.0 * x * y, 2.0 * r2 - 1.0, x ** 2 - y ** 2, 3.0 * x ** 2 * y - y ** 3])
#     zernb = jnp.array([(3.0 * r2 - 2.0) * y, (3.0 * r2 - 2.0) * x, x ** 3 - 3.0 * x * y ** 2, 4.0
#                        * x * y * (x ** 2 - y ** 2)])
#     zernc = jnp.array([(4.0 * r2 - 3.0) * zerna[0], 6.0 * r2 ** 2 - 6.0 * r2 + 1, (4.0 * r2 - 3.0) * zerna[2],
#                        x ** 4 - 6.0 * x ** 2 * y ** 2 + y ** 4])
#     return zerna, zernb, zernc
#
# @functools.partial(jax.jit, static_argnums=(3,))
# def compute_pixel(px, py, spots, n_spots):
#     xy = ((jnp.array([px, py]) / slm_resolution) - center) * hologram_size
#     phi = jnp.arctan2(xy[1], xy[0])
#     pos = jnp.array([xy[0] / f, xy[1] / f, 1.0 - jnp.dot(xy, xy) / (2.0 * f ** 2), phi / k])
#     x, y = (xy / hologram_size) * 2.0
#     zerna, zernb, zernc = calculate_zernike(x, y)
#
#     xs = jnp.arange(0, n_spots, 1).astype(jnp.int32)
#
#     def spot_f(i):
#         return jax.lax.dynamic_index_in_dim(spots, i, axis=0, keepdims=False)
#
#     def scan_func_ri(carry, x):
#         totalr, totali = carry
#         spot_data = spot_f(x)  # Get single spot data using dynamic slice
#         # jax.debug.print("spot_data: {spot_data} x: {x}", spot_data=spot_data, x=x)
#
#         amplitude = spot_data[1, 0]
#         # jax.debug.print("amplitude: {amplitude}", amplitude=amplitude)
#         phase = k * jnp.dot(spot_data[0, :], pos) + spot_data[1, 1]
#         na = spot_data[2, :3]
#
#         # Perform NA restriction
#         # if jnp.dot(na[:2] - xy / size, na[:2] - xy / size) > na[2] ** 2:
#         #    amplitude = 0.0
#
#         # Line trap logic
#         line = spot_data[3, :]
#         length = jnp.sqrt(jnp.dot(line[:3], line[:3]))
#         # if length > 0.0:
#         #    sx = k * jnp.dot(jnp.append(pos[:3], length), line)
#         #    if sx != 0.0:
#         #        amplitude *= jnp.sin(sx) / sx
#
#         # Convert amplitude + phase to real + imaginary
#         totalr += amplitude * jnp.sin(phase)
#         totali += amplitude * jnp.cos(phase)
#
#         carry = (totalr, totali)
#         return carry, (totalr, totali)
#
#     final, result = jax.lax.scan(scan_func_ri, (0, 0), xs)
#     totalr = final[0]  # result contains each cumulative sum as it steps through the for loop over number of spots
#     totali = final[1]
#
#     amplitude = jnp.sqrt(totalr ** 2 + totali ** 2)
#     phase = jnp.arctan2(totalr, totali)
#     # if amplitude == 0.0:
#     #    phase = 0.0
#     # if totalA > 0.0:
#     #    phase *= jnp.clip(amplitude / totalA, 0.0, 1.0)
#     phase += jnp.dot(zernikeCoefficients[0], zerna) + jnp.dot(zernikeCoefficients[1], zernb) + jnp.dot(
#         zernikeCoefficients[2], zernc)
#     phase = (phase + jnp.pi) % (2 * jnp.pi) - jnp.pi
#
#     # Convert phase to blazing table
#     phint = (jnp.floor((phase / (2 * jnp.pi) + 0.5) * (len(blazing) - 1))).astype(int)
#     alpha = (phase / (2 * jnp.pi) + 0.5) * (len(blazing) - 1) - phint
#     # pixel_value = white * blazing[phint] * (1.0 - alpha) + blazing[phint + 1] * alpha
#
#     return phase, amplitude
#
# @functools.partial(jax.jit, static_argnums=(2,))
# def compute_row(row, spots_params, n_spots):
#     cols = jnp.arange(slm_resolution)
#     return jax.vmap(lambda col: compute_pixel(row, col, spots_params, n_spots))(cols)
#
# @functools.partial(jax.jit, static_argnums=(1,))
# def calculate_phase_mask(spots_params, n_spots):
#     """
#     Parallelize computation of compute_pixel over all pixels
#     spots_params is a (n, 4, 4) array of spot parameters defining n spots
#     returns phase mask
#     """
#     rows = jnp.arange(slm_resolution)
#     return jax.vmap(lambda row: compute_row(row, spots_params, n_spots))(rows)
#
# @functools.partial(jax.jit, static_argnums=(1,))
# def batch_calculate_masks(spots_batch, n_spots):
#     """ (batch_size, n_spots, 4, 4) """
#     return jax.vmap(functools.partial(calculate_phase_mask, n_spots=n_spots))(spots_batch)
#
#
#
# if __name__ == "__main__":
#     spots_parameters = jnp.array([
#         [[6, -45.210987, 0.000000, 5.800000],
#          [1.100000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 1.150000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[4, -50.078036, 0.000000, 5.000000],
#          [0.600000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 1.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[1, -35.987654, 0.000000, 7.000000],
#          [0.400000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 1.100000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[2, -41.123456, 0.000000, 6.500000],
#          [0.400000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 1.200000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[1, -55.432198, 0.000000, 4.500000],
#          [1.050000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.900000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[90, -30.876543, 0.000000, 8.000000],
#          [0.950000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 1.300000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[4, -12.567890, 0.000000, 6.700000],
#          [0.870000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 1.250000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[90, -32.098765, 0.000000, 7.500000],
#          [0.920000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 1.050000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[90, -53.876543, 0.000000, 4.200000],
#          [1.030000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.950000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[90, -42.654321, 0.000000, 6.200000],
#          [0.880000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 1.180000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]]
#     ])
#
#     single_spot = np.array([
#
#         [[50, -45.210987, -8.000000, 0.000000],
#          [1.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 1.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[0, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[0, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[0, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[0, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[0, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[0, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[0, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[0, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]],
#
#         [[0, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000],
#          [0.000000, 0.000000, 0.000000, 0.000000]]
#     ])
#
#     mask = calculate_phase_mask(single_spot, 10)
#
#     plt.figure(figsize=(6, 6))
#     plt.imshow(mask[0], cmap='gray')  # Use 'gray' colormap for better visualization
#     plt.colorbar(label="Phase Value")
#     plt.title("Predicted Mask")
#     plt.axis("off")  # Remove axis for a cleaner look
#     plt.show()

