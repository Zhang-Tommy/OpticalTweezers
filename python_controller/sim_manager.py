
import numpy as np
from constants import *
from simulator import Bead
import random
import mpc

class SimManager:
    """ Defines functions for tracking, creating, moving, and deleting beads"""
    def __init__(self):
        # Holds beads in a 2d grid w/dimensions of camera
        self.grid = [[Bead() for _ in range(CAM_X)] for _ in range(CAM_Y)]
        self.trapped_beads = {}  # Holds x,y position (cam_coords) of trapped beads
        self.free_beads = {}
        self.num_beads = 0  # Number of active traps
        self.beads_vec = np.zeros(3200)  # Current trap data (sent to hologram engine)

    def move_randomly(self):
        """ """
        # for all non trapped beads, move randomly
        for bead in list(self.free_beads.values()):
            offset_x = random.randint(-1, 1)
            offset_y = random.randint(-1, 1)
            # Move bead by the random offsets

            bead.move_bead(bead.x + offset_x, bead.y + offset_y, (0, 0, 255))

    def brownian_move(self, dt, dynamics):
        """ Move all freely diffusing beads using Langevin equation"""
        for bead in list(self.free_beads.values()):
            x_next = bead.get_next_pos(dt, dynamics)

            bead.move_bead(bead.x + int(x_next[0]), bead.y + int(x_next[1]), (0, 0, 255))


    def get_spots(self):
        """ Updates all spot parameters (preparation for hologram engine update) """
        pass
        #spot_vals = np.zeros(16*self.num_spots)
        #count = 0

        #for bead in self.trapped_beads:
            #bead_params = self.grid[bead[0]][bead[1]].get_spot_params()
            #for i in range(16):
                #spot_vals[count] = bead_params[i]
                #count += 1

        #self.spots_vec = spot_vals

    def check_bounds(self, pos):
        if pos[0] < 0 or pos[1] < 0 or pos[0] > CAM_X or pos[1] > CAM_Y:
            print('trap out of bounds')
            return False
        else:
            return True

    def update_traps(self):
        """ Updates hologram engine with current spot parameter data """
        pass
        #self.get_spots()
        #start = '<uniform id=2>\n'
        #end = '\n</uniform>'

        # Format the numbers with 6 decimal places and join them with a single space
        #string = ' '.join(f'{val:.6f}' for val in self.spots_vec[0:self.num_spots * 16])

        #packet = start + string + end
        # Send to hologram engine
        #send_data(packet)

    def add_bead(self, pos):
        """Creates a new spot and sends it over to the hologram engine for rendering"""
        # Add the desired spot to the spot object
        new_bead = Bead(pos[0], pos[1])
        #new_spot.change_pos(cam_to_um(pos))
        #self.trapped_beads[pos] = new_bead

        self.num_beads += 1
        self.free_beads[pos] = new_bead
        # Add to the grid and update holo engine
        if not self.grid[pos[0]][pos[1]].active:
            self.grid[pos[0]][pos[1]] = new_bead
            self.grid[pos[0]][pos[1]].active = True
            self.update_traps()
        else:
            print(f'Bead already exists at {pos[0]}, {pos[1]}')
            pass

    def move_beads(self, old_pos, new_pos):
        # Get the target spot object
        bead = self.grid[old_pos[0]][old_pos[1]]
        self.grid[old_pos[0]][old_pos[1]].active = False

        self.free_beads.pop(old_pos)

        #spot.change_pos(cam_to_um(new_pos))

        self.free_beads[new_pos] = bead

        self.grid[new_pos[0]][new_pos[1]] = bead
        self.grid[new_pos[0]][new_pos[1]].active = True

        # Send to hologram engine
        self.update_traps()

    def move_trap(self, old_pos, new_pos):
        # Get the target spot object
        bead = self.grid[old_pos[0]][old_pos[1]]
        self.grid[old_pos[0]][old_pos[1]].is_trapped = False

        self.grid[new_pos[0]][new_pos[1]] = bead
        self.grid[new_pos[0]][new_pos[1]].is_trapped = True


    def trap_bead(self, trap_loc):
        # Check if a bead is close to the trap
        bead_loc = self.check_for_bead(trap_loc)

        # if the bead is already trapped
        if bead_loc is not None:  # if there is a bead close to trap
            self.grid[bead_loc[0]][bead_loc[1]].is_trapped = True
            self.grid[bead_loc[0]][bead_loc[1]].move_bead(trap_loc[0], trap_loc[1], (0, 255, 0))
            self.move_trap((bead_loc[0], bead_loc[1]), (trap_loc[0], trap_loc[1]))
            if bead_loc in self.free_beads:
                self.free_beads.pop(bead_loc)


    def check_for_bead(self, trap_loc):
        """ Check if the given trap is near a bead """
        x_trap = trap_loc[0]
        y_trap = trap_loc[1]
        search_radius = 10  # pixels

        x_left = x_trap - search_radius
        x_right = x_trap + search_radius
        y_top = y_trap - search_radius
        y_bottom = y_trap + search_radius

        if x_left < 0:
            x_left = 0
        if x_right > CAM_X:
            x_right = CAM_X
        if y_bottom > CAM_Y:
            y_bottom = CAM_Y
        if y_top < 0:
            y_top = 0

        for i in range(x_left, x_right):
            for j in range(y_top, y_bottom):
                if self.grid[i][j].active:
                    return i, j

        return None




