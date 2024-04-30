import multiprocessing
from multiprocessing import Process
import random
import cv2
import camera

from utilities import *
from sim_manager import SimManager
from simulator import white_bg
from spot_manager import SpotManager

def holo(trap_parent, kp_child):
    """ Controls hologram engine and creating traps """
    # Initialize the hologram engine (sends commands to spatial light modulator that generates holograms)
    init_holo_engine()

    # Initialize spot manager (create, move, modify optical traps)
    sm = SpotManager()
    # Test operation of spot manager
    x1 = 300
    y1 = 300

    x2 = 100
    y2 = 250

    sm.add_spot((x1, y1))
    sm.add_spot((x2, y2))
    cnt = 0

    for i in range(10000):
        traps = list(sm.trapped_beads.keys())
        trap_parent.send(traps)
        if kp_child.poll():
            kps = kp_child.recv()

        #if cnt == 10:
        #   for kp in kp_child.recv():
        #        sm.add_spot((round(kp[0]), round(kp[1])))

        time.sleep(0.1)
        sm.move_trap((x1, y1), (x1-2, y1-1))
        sm.move_trap((x2, y2), (x2 - 2, y2 + 1))
        x1 = x1 - 2
        y1 = y1 - 1
        x2 = x2 - 2
        y2 = y2 + 1
        cnt += 1

def simulator(trap_child, kp_parent):
    """ Controls the simulator visualization w/random bead distribution """
    sm = SimManager()
    number_of_beads = 10

    for _ in range(number_of_beads):
        x_start = random.randint(0, CAM_Y - 1)
        y_start = random.randint(0, CAM_X - 1)
        sm.add_bead((x_start, y_start))

    traps = []
    counter = 0
    for i in range(100000):
        if i % 10 == 0:  # Move each bead randomly
            sm.move_randomly()

        # Poll for updated trap data (non-blocking)
        if trap_child.poll():
            traps = trap_child.recv()

        for trap in traps:
            # if trap is near a bead, then trap it
            sm.trap_bead((trap[0], trap[1]))
            cv2.circle(white_bg, (trap[0], trap[1]), 3, (0, 255, 0), -1)

        key_points = camera.detect_beads(white_bg)
        cv2.drawKeypoints(white_bg, key_points, white_bg, (255, 0, 0))

        cv2.imshow("Optical Tweezers Simulator", white_bg)

        for trap in traps:
            cv2.circle(white_bg, (trap[0], trap[1]), 13, (255, 255, 255), -1)
        points = []
        for kp in key_points:
            points.append([kp.pt[0], kp.pt[1]])

        counter += 1

        if counter % 10 == 0:
            kp_parent.send(points)

        cv2.waitKey(10)

    cv2.destroyAllWindows()

def controller():
    """ Controller for obstacle avoidance and path planning """
    # What does controller need to know?
    # State of system (coordinates of all obstacles, trapped beads)
    # Outputs: control x,y for specific bead



def cam(frame_queue, kp_parent):
    """ Video playback for testing (likely not very useful anymore since we have simulator) """
    # We can simulate the video feed from the tweezers by playing back a recorded video
    vid = cv2.VideoCapture(r'.\testing_video1.mp4')
    counter = 0
    while vid.isOpened():
        ret, frame = vid.read()

        if ret:
            key_points = camera.detect_beads(frame)
            cv2.drawKeypoints(frame, key_points, frame, (0, 0, 255), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("Frame", frame)
            cv2.waitKey(10)

            points = []

            for kp in key_points:
                points.append([kp.pt[0], kp.pt[1]])
            counter += 1

            if counter % 10 == 0:
                kp_parent.send(points)

            #frame_queue.put(frame)
        else:
            break


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    frame_queue = multiprocessing.Queue()  # Queue for camera frames
    kp_parent, kp_child = multiprocessing.Pipe()  # Communicates keypoints between processes
    trap_parent, trap_child = multiprocessing.Pipe()  # Communicates trap positions between processes

    p1 = Process(target=holo, args=(trap_parent, kp_child))
    p2 = Process(target=simulator, args=(trap_child, kp_parent))
    #p3 = Process(target=cam, args=(frame_queue, kp_parent))

    p1.start()
    p2.start()
    #p3.start()

    p1.join()
    p2.join()
    #p3.join()

