import numpy as np
from constants import *

class Spot:
    """Defines attributes for a single spot for the hologram rendering engine"""
    # 60.125000 54.187500 7.006914 0.000000 1.000000 1.000000 0.000000 0.000000
    # 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    def __init__(self):
        self.spot_vec = np.zeros(16)
        self.spot_vec[4] = 1.0  # intensity
        self.spot_vec[10] = 1.0  # na.r
        self.spot_vec[2] = Z_OFFSET  # z-axis offset
        self.active = False
        self.is_obstacle = False
        self.is_line = False
        self.is_donut = False
        self.is_virtual = False
        self.angle = 0
        self.is_goal = False

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

    def set_line_params(self, length=LINE_TRAP_LENGTH, angle=LINE_TRAP_ANGLE):
        # element 3 line trapping x y z and phase gradient.  xyz define the size and angle of the line
        x_length = length * np.cos(angle)
        y_length = length * np.sin(angle)
        self.angle = angle
        #self.spot_vec[3] = 2 # vortex charge
        self.spot_vec[4] = 1.5 # intensity
        self.spot_vec[12] = x_length
        self.spot_vec[13] = y_length
        self.is_line = True

    def set_donut_params(self, l=ANNULAR_TRAP_VORTEX_CHARGE):
        self.spot_vec[3] = l
        self.is_donut = True
        self.spot_vec[2] = Z_OFFSET + DONUT_Z_OFFSET
        self.spot_vec[4] = 1
        #self.spot_vec[12] = 1
        #self.spot_vec[13] = 1
