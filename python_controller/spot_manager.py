from spot import Spot
import numpy as np
from utilities import *
from constants import *


class SpotManager:
    """ Defines functions for tracking, creating, moving, and deleting traps"""
    def __init__(self):
        # Holds spots in a 2d grid w/dimensions of camera
        self.grid = [[Spot() for _ in range(CAM_X)] for _ in range(CAM_Y)]
        self.trapped_beads = {}  # Holds x,y position (cam_coords) of trapped beads
        self.num_spots = 0  # Number of active traps
        self.spots_vec = np.zeros(3200)  # Current trap data (sent to hologram engine)

    def get_spots(self):
        """ Updates all spot parameters (preparation for hologram engine update) """
        spot_vals = np.zeros(16*self.num_spots)
        count = 0

        for bead in self.trapped_beads:
            bead_params = self.grid[bead[0]][bead[1]].get_spot_params()
            for i in range(16):
                spot_vals[count] = bead_params[i]
                count += 1

        self.spots_vec = spot_vals

    def check_bounds(self, pos):
        if pos[0] < 0 or pos[1] < 0 or pos[0] > CAM_X or pos[1] > CAM_Y:
            print('trap out of bounds')
            return False
        else:
            return True

    def update_traps(self):
        """ Updates hologram engine with current spot parameter data """
        self.get_spots()
        start = '<uniform id=2>\n'
        end = '\n</uniform>'

        # Format the numbers with 6 decimal places and join them with a single space
        string = ' '.join(f'{val:.6f}' for val in self.spots_vec[0:self.num_spots * 16])

        packet = start + string + end
        # Send to hologram engine
        send_data(packet)

    def add_spot(self, pos):
        """Creates a new spot and sends it over to the hologram engine for rendering"""
        # Add the desired spot to the spot object

        if self.check_bounds(pos):
            new_spot = Spot()
            new_spot.change_pos(cam_to_um(pos))
            self.trapped_beads[pos] = new_spot

            self.num_spots += 1
            print(f'{pos[0]} {pos[1]}')
            # Add to the grid and update holo engine
            if not self.grid[pos[0]][pos[1]].active:
                self.grid[pos[0]][pos[1]] = new_spot
                self.grid[pos[0]][pos[1]].active = True
                self.update_traps()
            else:
                print(f'Trap already exists at {pos[0]}, {pos[1]}')
                pass

    def move_trap(self, old_pos, new_pos):
        # Get the target spot object
        spot = self.grid[old_pos[0]][old_pos[1]]
        self.grid[old_pos[0]][old_pos[1]].active = False
        self.trapped_beads.pop(old_pos)
        spot.change_pos(cam_to_um(new_pos))

        self.trapped_beads[new_pos] = spot
        self.grid[new_pos[0]][new_pos[1]] = spot
        self.grid[new_pos[0]][new_pos[1]].active = True

        # Send to hologram engine
        self.update_traps()




