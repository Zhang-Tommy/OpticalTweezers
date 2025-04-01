import numpy as np

def read_cube_lut():
    with open("dependencies/slm3894_at1064_P8.lut", "r") as f:
        lines = f.readlines()

    data_lines = [line.strip() for line in lines]

    # Extract data and reshape it
    data = np.array([float(x) for line in data_lines[:] for x in line.split()])
    data = data.reshape((256, 2))
    lut_map = dict(data)
    return lut_map

def apply_lut(phase_mask, lut_map):
    """Applies the look-up table (LUT) to the phase mask."""
    lut_array = np.zeros(256)  # Ensure a mapping exists for all values from 0 to 255

    # Populate the LUT array with values from the dictionary
    for key, value in lut_map.items():
        lut_array[int(key)] = value

    # Apply LUT mapping using NumPy advanced indexing
    transformed_mask = lut_array[phase_mask.astype(int)]

    rgb_image = np.stack([transformed_mask] * 3, axis=-1)
    return transformed_mask