import numpy as np
import cv2
import random
import camera
from constants import *

global white_bg
import mpc

# Create a white background image for the simulation display
white_bg = np.ones((CAM_Y, CAM_X, 3), dtype=np.uint8) * 255


class Bead:
    """
    Represents a bead in the simulator display.
    Each bead can be drawn, moved, and can be trapped or free.
    """
    def __init__(self, x_start=None, y_start=None):
        self.is_trapped = False  # Whether the bead is currently trapped by laser power
        self.x = x_start
        self.y = y_start
        self.bead_size = BEAD_RADIUS  # Radius used for drawing the bead
        if x_start is not None:
            self.create_bead()  # Draw initial bead if coordinates given
        self.active = False
        self.velocity = [np.array([0, 0]).T]  # Velocity vector, initially zero
        self.position = []

        self.v_next = 0  # Placeholder for next velocity (unused here)
        self.x_next = 0  # Placeholder for next position (unused here)

    def move_bead(self, x_new, y_new, color):
        # Erase the bead's old position by drawing a white circle over it
        cv2.circle(white_bg, (self.x, self.y), self.bead_size, (255, 255, 255), -1)
        # Draw the bead at the new position with the specified color
        cv2.circle(white_bg, (x_new, y_new), self.bead_size, color, -1)
        # Update bead's current position
        self.x = x_new
        self.y = y_new

    def create_bead(self):
        # Draw a solid red circle at the bead's current position
        cv2.circle(white_bg, (self.x, self.y), self.bead_size, (0, 0, 255), -1)

    def move_randomly(self):
        # Move the bead randomly if it is not trapped
        if not self.is_trapped:
            offset_x = random.randint(-1, 1)
            offset_y = random.randint(-1, 1)
            # Move the bead by a small random offset and redraw it
            self.move_bead(self.x + offset_x, self.y + offset_y, (0, 0, 255))
            return [offset_x, offset_y]  # Return the random move applied
        else:
            # If trapped, the bead does not move
            return [self.x, self.y]

    def get_next_pos(self, dt, dynamics):
        # Calculate next position using the RK4 integrator for the bead's dynamics
        a_next = mpc.RK4IntegratorObs(dynamics, dt)(self.velocity[0], dt)
        v_next = dt * a_next + self.v_next
        x_next = dt * v_next + self.x_next
        return x_next


def generate_random_beads(n):
    # Generate a list of n beads at random positions within the camera frame
    beads = []
    for _ in range(n):
        x_start = random.randint(0, CAM_X - 1)
        y_start = random.randint(0, CAM_Y - 1)
        beads.append(Bead(x_start, y_start))
    return beads


if __name__ == "__main__":
    # Initialize a fresh white background for the simulation
    white_bg = np.ones((CAM_Y, CAM_X, 3), dtype=np.uint8) * 255
    # Generate 10 beads randomly positioned
    beads = generate_random_beads(10)

    for i in range(100000):
        # Every other frame, move each bead randomly if it is free
        for bead in beads:
            if i % 2 == 0:
                bead.move_randomly()

        # Display the current simulation frame with beads
        cv2.imshow("Optical Tweezers Simulator", white_bg)

        # Detect bead keypoints from the current image (for tracking or analysis)
        key_points = camera.detect_beads(white_bg)

        # Draw detected keypoints as blue circles on the image
        cv2.drawKeypoints(white_bg, key_points, white_bg, (255, 0, 0))

        # Wait briefly (10 ms) before updating the frame
        cv2.waitKey(10)

    # After the loop ends, close all OpenCV display windows
    cv2.destroyAllWindows()

