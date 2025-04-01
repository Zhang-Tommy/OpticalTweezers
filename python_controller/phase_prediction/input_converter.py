"""
Accept parameters for traps and convert to intensity distributions w/out fft

Take spot parameters and extract locations of each trap and trap type

From three files, we will store intensity for point trap, annular trap, line trap
"""
import numpy as np
import matplotlib.pyplot as plt

mask_size = 128
point = np.load("ref_intensity/point_trap_128.npy")
annular = np.load("ref_intensity/annular_trap_128.npy")
line = np.load("ref_intensity/line_trap_128.npy")

def add_mask_with_offset(ref_mask, new_mask, center):
    """
    Adds a new mask onto a reference mask at a given center position.

    Parameters:
        ref_mask (ndarray): The reference intensity mask (2D array).
        new_mask (ndarray): The mask to be added (2D array).
        center (tuple): (y, x) coordinates where the center of new_mask should be placed in ref_mask.

    Returns:
        ndarray: Updated reference mask with the new mask added.
    """
    ref_h, ref_w = ref_mask.shape
    new_h, new_w = new_mask.shape
    cx, cy = center  # Desired center position in reference mask

    # Compute top-left corner of new_mask in reference frame
    start_y = cy - new_h // 2
    start_x = cx - new_w // 2

    # Compute valid overlap region
    y1_new = max(0, -start_y)  # Clip top overlap
    x1_new = max(0, -start_x)  # Clip left overlap
    y2_new = min(new_h, ref_h - start_y)  # Clip bottom overlap
    x2_new = min(new_w, ref_w - start_x)  # Clip right overlap

    y1_ref = max(0, start_y)
    x1_ref = max(0, start_x)
    y2_ref = y1_ref + (y2_new - y1_new)
    x2_ref = x1_ref + (x2_new - x1_new)

    # Add only the overlapping portion
    ref_mask[y1_ref:y2_ref, x1_ref:x2_ref] += new_mask[y1_new:y2_new, x1_new:x2_new]

    return ref_mask

def add_trap(intensity, x, y, type):
    if type == 'p':
        return add_mask_with_offset(intensity, point, (x,y))
    elif type == 'a':
        return add_mask_with_offset(intensity, annular, (x, y))
    else:
        return add_mask_with_offset(intensity, line, (x, y))

def gen_input_intensity(spot_params):
    intensity = np.zeros((mask_size, mask_size))

    for spot in spot_params:
        x = int(spot[0, 0] * 0.25)
        y = int(-spot[0, 1] * 0.25)
        if x != 0 and y != 0:
            if spot[0, 3] != 0:
                type = 'a'
            elif spot[3, 0] != 0:
                type = 'l'
            else:
                type = 'p'

            x = int(spot[0, 0] * 0.25 * ((mask_size/2/((mask_size / 512) * 120))) + mask_size/2)
            y = int(-spot[0, 1] * 0.25 *((mask_size/2/((mask_size / 512) * 120))) + mask_size/2)
            intensity = add_trap(intensity, x, y, type)

    return intensity

# single_spot = np.array([
#
#         [[15, -15, -0.000000, 0.000000],
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
# intensity = gen_input_intensity(single_spot)

# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#
# ax[0].imshow(np.zeros((mask_size, mask_size)), cmap='gray', interpolation='nearest')
# ax[0].set_title("Reference Mask")
# ax[0].axis("off")
#
# ax[1].imshow(intensity, cmap='gray', interpolation='nearest')
# ax[1].set_title("Updated Mask with New Mask Added")
# ax[1].axis("off")
#
# plt.show()
