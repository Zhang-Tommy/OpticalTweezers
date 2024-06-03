import multiprocessing
from multiprocessing import Process
import random
import cv2
import jax
import keyboard

import camera
import time
import numpy as np

from mpc import *
# import time

from utilities import *
from sim_manager import SimManager
from simulator import white_bg
from spot_manager import SpotManager
from harvesters.core import Harvester

def holo(trap_parent, kp_child, controls_parent):
    """ Controls hologram engine and creating traps """
    # Initialize the hologram engine (sends commands to spatial light modulator that generates holograms)
    holo_process = init_holo_engine()

    # Initialize spot manager (create, move, modify optical traps)
    sm = SpotManager()
    time.sleep(2)

    traps = list(sm.trapped_beads.keys())
    trap_parent.send(traps)

    if kp_child.poll():
        kps = kp_child.recv()

    num_beads = len(kps) - 1

    # Choose first detected bead as our target bead
    x_start = float(kps[0][0])
    y_start = float(kps[0][1])

    # Trap the target bead
    sm.add_spot((int(x_start), int(y_start)))

    # Define start and goal states
    start_state = jnp.array([x_start, y_start])
    goal_position = jnp.array([50, 50])

    dt = 1e-3

    kpsarray = jnp.asarray(kps)
    kpsarray = kpsarray[kpsarray != start_state[0]]  # remove start state from keypoints
    kpsarray = kpsarray[kpsarray != start_state[1]]
    env = Environment.create(num_beads, kpsarray)

    T = 7500
    N = 500
    u_guess = gen_initial_traj(start_state, goal_position, N).T

    state = start_state
    dynamics = RK4Integrator(ContinuousTimeBeadDynamics(), dt)

    start_time = time.time()

    for k in range(T - 1):
        policy_start = time.time()
        control, (opt_states, opt_controls) = policy(state, goal_position, u_guess, env, dynamics, RunningCost, MPCTerminalCost, False, N)
        policy_end = time.time()
        #print(f'Policy took {policy_end - policy_start} seconds to run')
        print(control)
        traps = list(sm.trapped_beads.keys())
        #print(f'State: {state}, Control: {control}')
        sm.move_trap((int(state[0]), int(state[1])), (int(control[0]), int(control[1])))
        state = control  # The control is the position of the bead

        if k % 10 == 1:  # "Synchronized" with random bead movement
            trap_parent.send(traps)
            controls_parent.send(opt_controls)
            if kp_child.poll():
                kps = kp_child.recv()
                kpsarray = jnp.asarray(kps)
                num_beads = len(kps)
                kpsarray = kpsarray[1:]

                env = env.update(kpsarray, num_beads-1)

        time.sleep(.005)

        ## if close to goal, break
        dist_to_goal = np.sqrt((state[0] - goal_position[0])**2 + (state[1] - goal_position[1])**2)
        if dist_to_goal < 1:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'Goal Reached in {elapsed_time} seconds!')
            break

        if keyboard.is_pressed('q'):
            holo_process.terminate()
            break

def simulator(trap_child, kp_parent, controls_child):
    """ Controls the simulator visualization w/random bead distribution """
    sm = SimManager()
    number_of_beads = 50

    dt = 0.0015

    for _ in range(number_of_beads):
        x_start = random.randint(0, CAM_X - 1)
        y_start = random.randint(0, CAM_Y - 1)
        sm.add_bead((x_start, y_start))

    traps = []

    i = 0
    dynamics = ContinuousTimeObstacleDynamics()
    old_controls = []
    while True:
        if i % 10 == 0:  # Move each bead randomly
            sm.brownian_move(dt, dynamics)

        # Poll for updated trap data (non-blocking)
        if trap_child.poll():
            traps = trap_child.recv()

        for trap in traps:
            # if trap is near a bead, then trap it
            sm.trap_bead((trap[0], trap[1]))
            cv2.circle(white_bg, (trap[0], trap[1]), 3, (0, 255, 0), -1)

        if controls_child.poll():

            for cont in old_controls:
                # print(opt_controls)
                cv2.circle(white_bg, (int(cont[0]), int(cont[1])), 2, (255, 255, 255), -1)

            opt_controls = controls_child.recv()
            opt_controls = opt_controls.reshape(-1, 2)
            old_controls = opt_controls
            for cont in opt_controls:
                # print(opt_controls)
                cv2.circle(white_bg, (int(cont[0]), int(cont[1])), 2, (255, 0, 0), -1)

        key_points = camera.detect_beads(white_bg)
        cv2.drawKeypoints(white_bg, key_points, white_bg, (255, 0, 0))

        cv2.imshow("Optical Tweezers Simulator", white_bg)

        for trap in traps:
            cv2.circle(white_bg, (trap[0], trap[1]), 18, (255, 255, 255), -1)
        points = []
        for kp in key_points:
            points.append([kp.pt[0], kp.pt[1]])

        if i % 10 == 0:
            kp_parent.send(points)

        cv2.waitKey(4)  # Millisecond delay
        i += 1

        if keyboard.is_pressed('q'):
            os.system("taskkill /f /im  hologram_engine_64.exe")
            break

    cv2.destroyAllWindows()


def cam(kp_parent):
    h = Harvester()
    h.add_file(r'C:\Users\User\Desktop\Tommy_Tweezers_Automation\tweezers_automation\tweezers_automation_v2\bgapi2_gige.cti')
    h.update()

    ia = h.create()
    ia.start()
    traps = []
    k = 0
    while True:
        with ia.fetch() as buffer:
            component = buffer.payload.components[0]
            img = np.ndarray(buffer=component.data.copy(), dtype=np.uint8,
                             shape=(component.height, component.width, 1))
            #img = np.repeat(img, 3, axis=2)
            key_points = camera.detect_beads(img)
            points = []

            if trap_child.poll():
                traps = trap_child.recv()
            for trap in traps:
                cv2.circle(img, (trap[0], trap[1]), 3, (255, 255, 255), -1)
            for kp in key_points:
                points.append([kp.pt[0], kp.pt[1]])
            if k % 10 == 1:
                kp_parent.send(points)

            cv2.waitKey(10)
            cv2.imshow('Camera Feed', img)
            k += 1

        if keyboard.is_pressed('q'):
            ia.stop()
            ia.destroy()
            os.system("taskkill /f /im  hologram_engine_64.exe")
            break

def exit():
    # close hologram engine
    # close opencv window
    # close camera connection
    # close any open sockets
    pass

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    frame_queue = multiprocessing.Queue()  # Queue for camera frames
    kp_parent, kp_child = multiprocessing.Pipe()  # Communicates keypoints between processes
    trap_parent, trap_child = multiprocessing.Pipe()  # Communicates trap positions between processes
    controls_parent, controls_child = multiprocessing.Pipe() # Communicates future optimal controls to simulator

    p1 = Process(target=holo, args=(trap_parent, kp_child, controls_parent))
    p2 = Process(target=simulator, args=(trap_child, kp_parent, controls_child))
    #p3 = Process(target=cam, args=(kp_parent,))

    p2.start()
    p1.start()
    #p3.start()

    p1.join()
    p2.join()
    #p3.join()
