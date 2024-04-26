import numpy as np
from utilities import update_spots


class Spot:
    """Defines attributes for a single spot for the hologram rendering engine"""
    # 60.125000 54.187500 7.006914 0.000000 1.000000 1.000000 0.000000 0.000000
    # 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    def __init__(self):
        self.spot_vec = np.zeros(16)
        self.spot_vec[4] = 1.0  # intensity
        self.spot_vec[10] = 1.0  # na.r
        self.active = False

    def change_pos(self, pos):
        """Changes position of the spot rendered"""
        # Elements 0, 1, 2, 3 -> x y z l (x,y,z in um, l is integer)
        self.spot_vec[0] = pos[0]
        self.spot_vec[1] = -pos[1]

    def change_intensity(self, intensity):
        """Changes intensity of the spot rendered"""
        # Elements 4, 5, 6, 7 -> intensity (I), phase, -, -
        self.spot_vec[4] = intensity

    def change_phase(self, phase):
        """Changes phase of the spot rendered"""
        # Elements 4, 5, 6, 7 -> intensity (I), phase, -, -
        self.spot_vec[5] = phase

    def get_spot_params(self):
        return self.spot_vec
