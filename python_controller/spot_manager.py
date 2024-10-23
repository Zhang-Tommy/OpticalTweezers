from spot import Spot
import numpy as np
from utilities import *
from constants import *


class SpotManager:
    """ Defines functions for tracking, creating, moving, and deleting traps"""
    def __init__(self):
        # Holds spots in a 2d grid w/dimensions of camera
        self.grid = [[Spot() for _ in range(CAM_Y)] for _ in range(CAM_X)]
        self.trapped_beads = {}  # Holds x,y position (cam_coords) of trapped beads
        self.virtual_traps = {}  # Holds positions of virtual traps (used to control laser power)
        self.num_spots = 0  # Number of active traps
        self.spots_vec = np.zeros(3200)  # Current trap data (sent to hologram engine)
        self.goal_positions = {}  # holds the user input goal positions
        self.obstacles = []  # keys are (x,y) pos of obstacles, value is keypoint object
        self.virtual_x_pos = np.arange(20)
        self.clearing_region = False

    def set_clearing_region(self, bool):
        self.clearing_region = bool

    def get_clearing_region(self):
        return self.clearing_region

    def set_obstacles(self, new_obstacles):
        self.obstacles = new_obstacles

    def get_obstacles(self):
        obstacles_with_traps = self.obstacles
        #print(list(self.trapped_beads.keys()))
        for pos in self.trapped_beads.keys():
            obstacles_with_traps.append(pos)
        return obstacles_with_traps

    def get_trapped_beads(self):
        """
        Returns dictionary of trapped beads {(x,y): SpotObject}
        """
        return self.trapped_beads

    def get_virtual_traps(self):
        return self.virtual_traps

    def get_goal_pos(self):
        return self.goal_positions

    def add_goal_pos(self, pos):
        self.goal_positions[pos] = None
        self.grid[pos[0]][pos[1]].is_goal = True

    def remove_goal_pos(self, pos):
        self.goal_positions.pop(pos)
        self.grid[pos[0]][pos[1]].is_goal = False

    def get_spots(self):
        """
        Updates all spot parameters (preparation for hologram engine update)
        """
        spot_vals = np.zeros(16*self.num_spots)
        count = 0

        if self.num_spots > 0:
            for bead in self.trapped_beads:
                bead_params = self.grid[bead[0]][bead[1]].get_spot_params()
                for i in range(16):
                    spot_vals[count] = bead_params[i]
                    count += 1

            for bead in self.virtual_traps:
                bead_params = self.grid[bead[0]][bead[1]].get_spot_params()
                for i in range(16):
                    spot_vals[count] = bead_params[i]
                    count += 1

            self.spots_vec = spot_vals
        else:
            self.spots_vec = np.zeros(3200)

    def check_bounds(self, pos):
        """
        Ensure desired positions for traps are within boundaries of workspace
        @param pos: x, y position in camera coords
        """
        if pos[0] < 0 or pos[1] < 0 or pos[0] > CAM_X or pos[1] > CAM_Y:
            print('trap out of bounds')
            return False
        else:
            return True

    def update_traps(self):
        """
        Updates hologram engine with current spot parameter data
        """
        self.get_spots()
        start = '<uniform id=2>\n'
        end = '\n</uniform>'

        # Format the numbers with 6 decimal places and join them with a single space
        string = ' '.join(f'{val:.6f}' for val in self.spots_vec[0:self.num_spots * 16])
        self.update_num_traps()
        packet = start + string + end

        # Send to hologram engine
        send_data(packet)

    def update_num_traps(self):
        start = '<uniform id=0>\n'
        end = '\n</uniform>'

        string = f'{self.num_spots:.6f}'
        packet = start + string + end
        send_data(packet)

    def add_spot(self, pos, is_line=False, is_donut=False, is_virtual=False):
        new_pos_scaled = self.offset_misalignment(pos)

        if self.check_bounds(pos):
            new_spot = Spot()
            new_spot.change_pos(cam_to_um(new_pos_scaled))
            if is_virtual:
                self.virtual_traps[pos] = new_spot
            else:
                self.trapped_beads[pos] = new_spot
            self.num_spots += 1
            if is_line:
                new_spot.set_line_params()
            elif is_donut:
                new_spot.set_donut_params()
            # Add to the grid and update holo engine
            if not self.grid[pos[0]][pos[1]].active:
                self.grid[pos[0]][pos[1]] = new_spot
                self.grid[pos[0]][pos[1]].active = True
                if not is_virtual:
                    self.update_virtual_traps()
                self.update_traps()
            else:
                print(f'Trap already exists at {pos[0]}, {pos[1]}')
                pass

    def move_trap(self, old_pos, new_pos):
        """
        Moves the spot trap from current position to desired position
        @param old_pos: current position in camera coords
        @param new_pos: desired position in camera coords
        """
        pos = new_pos
        new_pos_scaled = self.offset_misalignment(pos)
        #print(new_pos_scaled)
        # Get the target spot object
        spot = self.grid[old_pos[0]][old_pos[1]]
        self.grid[old_pos[0]][old_pos[1]].active = False
        self.trapped_beads.pop(old_pos)
        spot.change_pos(cam_to_um(new_pos_scaled))

        self.trapped_beads[new_pos] = spot
        self.grid[new_pos[0]][new_pos[1]] = spot
        self.grid[new_pos[0]][new_pos[1]].active = True

        # Send to hologram engine
        self.update_traps()

    def remove_trap(self, pos, is_virtual=False):
        new_spot = Spot()
        new_spot.spot_vec = np.zeros(16)
        if not is_virtual:
            self.update_virtual_traps()
            self.trapped_beads.pop(pos)

        self.grid[pos[0]][pos[1]] = new_spot
        self.num_spots -= 1

        self.update_traps()

    def offset_misalignment(self, pos):
        x, y = pos
        x_rotated = x * np.cos(ANGLE) - y * np.sin(ANGLE)
        y_rotated = x * np.sin(ANGLE) + y * np.cos(ANGLE)

        offset_x = x_rotated * SCALE_X
        offset_y = y_rotated * SCALE_Y

        return offset_y, offset_x

    def update_virtual_traps(self):
        """
        Ensures traps have a similar amount of power
        """
        num_trapped_beads = len(self.trapped_beads)
        if num_trapped_beads == 0:
            return
        num_virtual_traps = len(self.virtual_traps)

        total_power = num_virtual_traps * AVG_DESIRED_LASER_PWR + num_trapped_beads * AVG_DESIRED_LASER_PWR
        #print(f"Total Power: {total_power}, Num_Virtual_Traps={num_virtual_traps}, Num_Trapped_beads={num_trapped_beads}")
        if total_power > TOTAL_LASER_PWR:
            num_virtual_remove = int((total_power - TOTAL_LASER_PWR) / AVG_DESIRED_LASER_PWR)
            if num_virtual_traps == 0:
                print("Warning: Cannot remove more virtual traps - trap power will be limited")
            else:
                for i in range(num_virtual_remove):
                    virtual_trap = self.virtual_traps.popitem()  # remove virtual trap
                    self.remove_trap(virtual_trap[0], is_virtual=True)
        elif total_power < TOTAL_LASER_PWR:
            num_virtual_add = int((TOTAL_LASER_PWR - total_power) / AVG_DESIRED_LASER_PWR)
            for i in range(num_virtual_add):
                x = self.virtual_x_pos[num_virtual_traps + 1 + i]
                self.add_spot((x, 0), is_virtual=True)
