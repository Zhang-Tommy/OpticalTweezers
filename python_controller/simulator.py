import numpy as np
import cv2
import random
import camera
global white_bg

width, height = 640, 480  # Dimensions of the image
white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255

bead_size = 10  # diameter in pixels


class Bead:
    def __init__(self, x_start, y_start):
        self.is_trapped = False
        self.x = x_start
        self.y = y_start
        self.create_bead()

    def move_bead(self, x_new, y_new):
        cv2.circle(white_bg, (self.x, self.y), bead_size, (255, 255, 255), -1)
        cv2.circle(white_bg, (x_new, y_new), bead_size, (0, 0, 255), -1)
        self.x = x_new
        self.y = y_new

    def create_bead(self):
        cv2.circle(white_bg, (self.x, self.y), bead_size, (0, 0, 255), -1)

    def move_randomly(self):
        if not self.is_trapped:
            offset_x = random.randint(-1, 1)
            offset_y = random.randint(-1, 1)
            # Move bead by the random offsets
            self.move_bead(self.x + offset_x, self.y + offset_y)
        else:
            pass



def generate_random_beads(n):
    beads = []
    for _ in range(n):
        x_start = random.randint(0, width - 1)
        y_start = random.randint(0, height - 1)
        beads.append(Bead(x_start, y_start))
    return beads

if __name__ == "__main__":
    beads = generate_random_beads(10)

    for i in range(100000):
        # Call the function to draw a red circle

        for bead in beads:
            if i % 2 == 0:
                bead.move_randomly()  # Move each bead randomly


        beads[0].is_trapped = True
        # Display the image with the red circle
        cv2.imshow("Optical Tweezers Simulator", white_bg)
        #print(camera.detect_beads(white_bg))
        cv2.waitKey(10)  # Wait for any key press before closing

    # Close all OpenCV windows
    cv2.destroyAllWindows()

