from constants import *
from simulator import Bead
import random


class SimManager:
    """ Defines functions for tracking, creating, moving, and deleting beads in the simulator"""
    def __init__(self):
        # Holds beads in a 2d grid w/dimensions of camera
        self.grid = [[Bead() for _ in range(CAM_Y)] for _ in range(CAM_X)]
        self.trapped_beads = {}  # Holds x,y position (cam_coords) of trapped beads
        self.free_beads = {}
        self.num_beads = 0  # Number of beads in workspace

    def move_randomly(self):
        """ Random walk bead movement """
        for bead in list(self.free_beads.values()):
            offset_x = random.randint(-1, 1)
            offset_y = random.randint(-1, 1)
            bead.move_bead(bead.x + offset_x, bead.y + offset_y, (0, 0, 255))

    def brownian_move(self, dt, dynamics):
        """ Move all freely diffusing beads using Langevin equation """
        for bead in list(self.free_beads.values()):
            x_next = bead.get_next_pos(dt, dynamics)
            bead.move_bead(bead.x + int(x_next[0]), bead.y + int(x_next[1]), (0, 0, 255))

    def add_bead(self, pos):
        """Creates a new spot and sends it over to the hologram engine for rendering"""
        # Add the desired spot to the spot object
        new_bead = Bead(pos[0], pos[1])
        self.num_beads += 1
        self.free_beads[pos] = new_bead

        # Add to the grid and update holo engine
        if not self.grid[pos[0]][pos[1]].active:
            self.grid[pos[0]][pos[1]] = new_bead
            self.grid[pos[0]][pos[1]].active = True
        else:
            print(f'Bead already exists at {pos[0]}, {pos[1]}')
            pass

    def move_beads(self, old_pos, new_pos):
        # Get the target spot object
        bead = self.grid[old_pos[0]][old_pos[1]]
        self.grid[old_pos[0]][old_pos[1]].active = False
        self.free_beads.pop(old_pos)
        self.free_beads[new_pos] = bead
        self.grid[new_pos[0]][new_pos[1]] = bead
        self.grid[new_pos[0]][new_pos[1]].active = True

    def move_trap(self, old_pos, new_pos):
        """ Move trap from previous position to desired position """
        bead = self.grid[old_pos[0]][old_pos[1]]
        self.grid[old_pos[0]][old_pos[1]].is_trapped = False
        self.grid[new_pos[0]][new_pos[1]] = bead
        self.grid[new_pos[0]][new_pos[1]].is_trapped = True

    def trap_bead(self, trap_loc):
        """ Use trap locations to trap simulator beads """
        bead_loc = self.check_for_bead(trap_loc)

        if bead_loc is not None:
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




