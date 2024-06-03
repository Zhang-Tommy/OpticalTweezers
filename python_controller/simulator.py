import numpy as np
import cv2
import random
import camera
from constants import *

global white_bg
import mpc

white_bg = np.ones((CAM_Y, CAM_X, 3), dtype = np.uint8) * 255


class Bead:
    def __init__(self, x_start=None, y_start=None):
        self.is_trapped = False
        self.x = x_start
        self.y = y_start
        self.bead_size = 12
        if x_start is not None:
            self.create_bead()
        self.active = False
        self.velocity = [np.array([0, 0]).T]
        self.position = []

        self.v_next = 0
        self.x_next = 0

    def move_bead(self, x_new, y_new, color):
        # Draw previous bead as white bead
        cv2.circle(white_bg, (self.x, self.y), self.bead_size, (255, 255, 255), -1)
        # Draw new bead position with correct color
        cv2.circle(white_bg, (x_new, y_new), self.bead_size, color, -1)
        self.x = x_new
        self.y = y_new

    def create_bead(self):
        # Draw a circle with specified position and color
        cv2.circle(white_bg, (self.x, self.y), self.bead_size, (0, 0, 255), -1)

    def move_randomly(self):
        if not self.is_trapped:
            offset_x = random.randint(-1, 1)
            offset_y = random.randint(-1, 1)
            # Move bead by the random offsets
            self.move_bead(self.x + offset_x, self.y + offset_y, (0, 0, 255))
            return [offset_x, offset_y]
        else:
            return [self.x, self.y]

    def get_next_pos(self, dt, dynamics):
        a_next = mpc.RK4IntegratorObs(dynamics, dt)(self.velocity[0], dt)
        v_next = dt * a_next + self.v_next
        x_next = dt * v_next + self.x_next
        return x_next


def generate_random_beads(n):
    beads = []
    for _ in range(n):
        x_start = random.randint(0, CAM_X - 1)
        y_start = random.randint(0, CAM_Y - 1)
        beads.append(Bead(x_start, y_start))
    return beads


if __name__ == "__main__":
    white_bg = np.ones((CAM_Y, CAM_X, 3), dtype = np.uint8) * 255
    beads = generate_random_beads(10)

    for i in range(100000):
        # Call the function to draw a red circle

        for bead in beads:
            if i % 2 == 0:
                bead.move_randomly()  # Move each bead randomly

        # Display the image with the red circle
        cv2.imshow("Optical Tweezers Simulator", white_bg)
        key_points = camera.detect_beads(white_bg)

        cv2.drawKeypoints(white_bg, key_points, white_bg, (255, 0, 0))
        cv2.waitKey(10)  # Wait for any key press before closing

    # Close all OpenCV windows
    cv2.destroyAllWindows()
