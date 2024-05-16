import multiprocessing
from multiprocessing import Process
import random
import cv2
import jax

import camera
import time
import numpy as np

from beads import *
# import time

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

    #simulator_mpc(sm) # Do full horizon ilqr

    traps = list(sm.trapped_beads.keys())

    trap_parent.send(traps)
    if kp_child.poll():
        kps = kp_child.recv()

    num_beads = len(kps)

    x_start = float(round(kps[0][0]))
    y_start = float(round(kps[0][1]))

    sm.add_spot((int(x_start), int(y_start)))

    start_state = jnp.array([x_start, y_start])
    goal_position = jnp.array([0, 0]) #Choose goal state
    # Get ideal trajectory w/out obstacles to init ilqr


    dt = 1e-6
    kpsarray = jnp.asarray(kps)
    env = Environment.create(num_beads, kpsarray)

    T = 7500
    N = 25
    u_guess = gen_intial_traj(start_state, goal_position, N).T
    #states = controller(start_state, goal_position, u_guess, env, T, dt)
    state = start_state
    dynamics = RK4Integrator(ContinuousTimeBeadDynamics(), dt)
    for k in range(T - 1):
        start_time = time.time()
        control, (mpc_states, mpc_controls) = policy(state, goal_position, u_guess, env, dynamics, RunningCost, FullHorizonTerminalCost, False, N)
        #print("--- %s seconds ---" % (time.time() - start_time))
        traps = list(sm.trapped_beads.keys())


        #print(f'State: {state}, Control: {control}')
        sm.move_trap((int(state[0]), int(state[1])), (int(control[0]), int(control[1])))
        state = control
        trap_parent.send(traps)
        if k % 30 == 1:
            if kp_child.poll():
                kps = kp_child.recv()
                kpsarray = jnp.asarray(kps)
                num_beads = len(kps)
            env = Environment.create(num_beads, kpsarray)
        time.sleep(.01)

def simulator(trap_child, kp_parent):
    """ Controls the simulator visualization w/random bead distribution """
    sm = SimManager()
    number_of_beads = 50

    for _ in range(number_of_beads):
        x_start = random.randint(0, CAM_Y - 1)
        y_start = random.randint(0, CAM_X - 1)
        sm.add_bead((x_start, y_start))

    traps = []
    sm.add_bead((0, 0))
    sm.trap_bead((0, 0))
    # Hardcoded 1999 for now (=T-1)
    i = 0
    while True:
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

        if i % 30 == 0:
            kp_parent.send(points)

        cv2.waitKey(10)
        i+=1

    cv2.destroyAllWindows()


def controller(start_state, goal_position, u_guess, env, T, dt):
    """ Controller for obstacle avoidance and path planning """
    # What does controller need to know?
    # State of system (coordinates of all obstacles, trapped beads)
    # Outputs: control x,y for specific bead

    dynamics = RK4Integrator(ContinuousTimeBeadDynamics(), dt)
    
    states, controls = simulate_mpc(start_state, goal_position, u_guess, env, dynamics, RunningCost, FullHorizonTerminalCost, False, 20, T)


    return states
    
def simulator_mpc(sm):
    traps = list(sm.trapped_beads.keys())

    trap_parent.send(traps)
    if kp_child.poll():
        kps = kp_child.recv()

    num_beads = len(kps)

    x = float(round(kps[0][0]))
    y = float(round(kps[0][1]))

    sm.add_spot((int(x), int(y)))

    T = 7500

    start_state = jnp.array([x, y])
    print(start_state)
    goal_position = jnp.array([0, 0])

    u_guess = gen_intial_traj(start_state, goal_position, 20).T

    dt = 1e-6
    kpsarray = jnp.asarray(kps)
    # kpsarray = kpsarray.at[:, 1].set(480 - kpsarray[:, 1])

    # print(kpsarray)
    env = Environment.create(num_beads, kpsarray)

    states = controller(start_state, goal_position, u_guess, env, T, dt)

    for k in range(T - 1):
        traps = list(sm.trapped_beads.keys())

        trap_parent.send(traps)
        if kp_child.poll():
            kps = kp_child.recv()

        x_curr = int(states[k][0])
        y_curr = int(states[k][1])

        x_next = int(states[k + 1][0])
        y_next = int(states[k + 1][1])
        # print(f'X:{x_next}  Y:{y_next}')
        sm.move_trap((x_curr, y_curr), (x_next, y_next))

        time.sleep(0.01)

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

    

    p2.start()
    p1.start()
    #p3.start()

    p1.join()
    p2.join()
    #p3.join()

