import multiprocessing
from multiprocessing import Process, Lock, Manager
from multiprocessing.managers import BaseManager
import threading
import random
from turtledemo.penrose import start

import cv2
import jax
import keyboard

import camera
import time
import numpy as np
import timeit

from mpc import *
# import time

from utilities import *
from sim_manager import SimManager
from simulator import white_bg
from spot_manager import SpotManager
from harvesters.core import Harvester

""" Doesn't really work, should remove the controlled bead from obstacles"""
def remove_closest_point(coords, x, y):
    # Calculate the Euclidean distance between each point and the given (x, y)
    distances = np.sqrt((coords[:, 0] - x) ** 2 + (coords[:, 1] - y) ** 2)

    # Find the index of the point with the minimum distance
    closest_index = np.argmin(distances)
    closest_dist = distances[closest_index]

    if closest_dist > 20:
        #print("too far away")
        coords = coords[1:]
        return coords

    # Remove the closest point by deleting that index
    updated_coords = np.delete(coords, closest_index, axis=0)
    #print(coords[closest_index])
    #print(f"{x}, {y}")
    return updated_coords

def holo(lock, spot_lock, trap_parent, kp_child, controls_parent, target_bead, spot_man):
    """ Controls hologram engine and creating traps """
    # Initialize the hologram engine (sends commands to spatial light modulator that generates holograms)
    #holo_process = init_holo_engine()

    # Initialize spot manager (create, move, modify optical traps)
    #spot_man = SpotManager()  # using two of these may not be optimal because of the data being sent to hologram engine

    """"""
    spot_lock.acquire()
    #
    #trap = np.array(list(spot_man.get_trapped_beads().keys()))

    lock.acquire()
    #trap_parent.send(trap)

    kps = kp_child.recv()

    num_beads = len(kps) - 1

    # Choose first detected bead as our target bead
    x_start = float(kps[target_bead][0])
    y_start = float(kps[target_bead][1])

    # Trap the target bead
    spot_man.add_spot((int(x_start), int(y_start)))
    spot_lock.release()
    # Define start and goal states
    start_state = jnp.array([x_start, y_start])
    goal_position = jnp.array([50, 50])

    dt = 1e-3

    kpsarray = jnp.asarray(kps)
    kpsarray = kpsarray[kpsarray != start_state[0]]  # remove start state from keypoints
    kpsarray = kpsarray[kpsarray != start_state[1]]
    #print(len(kpsarray))
    env = Environment.create(num_beads, kpsarray)

    T = 7500
    N = 125
    u_guess = gen_initial_traj(start_state, goal_position, N).T

    state = start_state
    dynamics = RK4Integrator(ContinuousTimeBeadDynamics(), dt)
    lock.release()
    #traps = np.zeros((MAX_NUM_TRAPS, 2), dtype=int)

    start_time = time.time()
    # the MPC loop runs at 30 hz
    for k in range(T - 1):
        st = time.time()
        empty_env = Environment.create(0, jnp.array([]))
        solution = policy(state, goal_position, u_guess, env, dynamics, RunningCost, MPCTerminalCost, empty_env, False, N)
        states, opt_controls = solution["optimal_trajectory"]
        control = opt_controls[0]
        spot_lock.acquire()
        try:
            spot_man.move_trap((int(state[0]), int(state[1])), (int(control[0]), int(control[1])))
        except:
            print(f"Trap move out of bounds invalid: {state[0]}, {state[1]} to {control[0]}, {control[1]}")

        #traps = list(spot_man.trapped_beads.keys())
        #print(spot_man.trapped_beads.keys())

        traps = list(spot_man.get_trapped_beads().keys())
        spot_lock.release()
        #print(trap)
        prev_state = state
        state = control  # The control is the position of the bead

        lock.acquire()
        trap_parent.send(traps) # can just try with two, make non-scalable
        controls_parent.send(opt_controls)

        if kp_child.poll():
            kps = kp_child.recv()
            kpsarray = jnp.asarray(kps)
            num_beads = len(kps)
            # we need to ensure the bead we are moving is removed from this
            kpsarray = remove_closest_point(kpsarray, prev_state[0], prev_state[1])
            #kpsarray = kpsarray[1:]

            env = env.update(kpsarray, num_beads-1)
        lock.release()
        time.sleep(.015)

        ## if close to goal, break
        dist_to_goal = np.sqrt((state[0] - goal_position[0])**2 + (state[1] - goal_position[1])**2)
        if dist_to_goal < 5:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'Goal Reached in {elapsed_time} seconds!')
            break

        if keyboard.is_pressed('q'):
            #holo_process.terminate()
            break
        et = time.time()

        # rudimentary timing controller
        if 1 / (et - st) > 20 and 0.05 - (et - st) > 0:
            time.sleep(0.05 - (et - st))

def simulator(lock, spot_lock, trap_child, kp_parent, controls_child):
    """ Controls the simulator visualization w/random bead distribution """
    # runs at about 12.5 hz
    sim_man = SimManager()
    number_of_beads = 5

    dt = 0.0015

    for _ in range(number_of_beads):
        x_start = random.randint(0, CAM_X - 1)
        y_start = random.randint(0, CAM_Y - 1)
        sim_man.add_bead((x_start, y_start))

    traps = []

    i = 0
    dynamics = ContinuousTimeObstacleDynamics()
    old_controls = []
    while True:
        if i % 1 == 0:  # Move each bead randomly
            sim_man.brownian_move(dt, dynamics)

        # Poll for updated trap data (non-blocking)

        if trap_child.poll():
            traps = trap_child.recv()
            print(traps)

        for trap in traps:
            # if trap is near a bead, then trap it
            #sim_man.trap_bead((trap[0], trap[1]))
            cv2.circle(white_bg, (trap[0], trap[1]), 10, (0, 255, 0), -1)


        if controls_child.poll():
            #print("Controls received")
            for t, cont in enumerate(old_controls):
                if t % 25 == 0:
                    cv2.circle(white_bg, (int(cont[0]), int(cont[1])), 2, (255, 255, 255), -1)

            opt_controls = controls_child.recv()
            opt_controls = opt_controls.reshape(-1, 2)
            old_controls = opt_controls

            for g, cont in enumerate(opt_controls):
                if g % 25 == 0:
                    cv2.circle(white_bg, (int(cont[0]), int(cont[1])), 2, (255, 0, 0), -1)

        key_points = camera.detect_beads(white_bg)
        cv2.drawKeypoints(white_bg, key_points, white_bg, (255, 0, 0))

        cv2.imshow("Optical Tweezers Simulator", white_bg)

        for trap in traps:
            cv2.circle(white_bg, (trap[0], trap[1]), 18, (255, 255, 255), -1)
        points = []

        for kp in key_points:
            points.append([kp.pt[0], kp.pt[1]])

        if len(points) < (number_of_beads - 1):
            diff = number_of_beads - len(points) - 1
            for e in range(diff):
                points.append([float(e), 0.0])

        # check if points has same number of elements as num_beads, if not pad with zeros

        #if i % 10 == 0:
            #kp_parent.send(points)

        threading.Thread(target=kp_parent.send, args=(points,)).start()

        cv2.waitKey(1)  # Millisecond delay
        i += 1

        if keyboard.is_pressed('q'):
            os.system("taskkill /f /im  hologram_engine_64.exe")
            break

    cv2.destroyAllWindows()

""" trap_parent.send(traps) # can just try with two, make non-scalable
    controls_parent.send(opt_controls)"""
def sync_pipes(one, two, parent):
    # wait for both one and two to send something
    one_recvd = False
    two_recvd = False
    while True:
        if one.poll():
            trap_one = one.recv()
            print(trap_one)
            one_recvd = True

        if two.poll():
            trap_two = two.recv()
            two_recvd = True

        if one_recvd and two_recvd:
            traps = trap_one.append(trap_two)
            parent.send(traps)
            print(traps)
            one_recvd = False
            two_recvd = False
        elif one_recvd:
            two_recvd = False
        else:
            one_recvd = False

def cam(kp_parent):
    ia = start_image_acquisition()

    k = 0
    while True:
        with ia.fetch() as buffer:
            component = buffer.payload.components[0]
            img = np.ndarray(buffer=component.data.copy(), dtype=np.uint8,
                             shape=(component.height, component.width, 1))

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

class SpotManagerManager(BaseManager):
    pass

if __name__ == "__main__":
    SpotManagerManager.register('SpotManager', SpotManager)
    SpotManagerManager.register('SpotManager', SpotManager)
    SpotManagerManager.register('get_trapped_beads', SpotManager.get_trapped_beads)
    SpotManagerManager.register('add_spot', SpotManager.add_spot)
    SpotManagerManager.register('move_trap', SpotManager.move_trap)

    manager = SpotManagerManager()
    manager.start()
    frame_queue = multiprocessing.Queue()  # Queue for camera frames
    kp_parent, kp_child = multiprocessing.Pipe()  # Communicates keypoints between processes
    #trap_parent_one, trap_child_one = multiprocessing.Pipe()  # Communicates trap positions between processes
    #trap_parent_two, trap_child_two = multiprocessing.Pipe()  # Communicates trap positions between processes
    trap_parent, trap_child = multiprocessing.Pipe()  # Communicates trap positions between processes
    controls_parent, controls_child = multiprocessing.Pipe() # Communicates future optimal controls to simulator
    #controls_parent_one, controls_child_one = multiprocessing.Pipe()  # Communicates controls between processes
    #controls_parent_two, controls_child_two = multiprocessing.Pipe()  # Communicates controls between processes

    spot_man = manager.SpotManager()
    lock = Lock()
    spot_lock = Lock()

    #p0 = Process(target=sync_pipes, args=(trap_child_one, trap_child_two, trap_parent))
    p1 = Process(target=holo, args=(lock, spot_lock, trap_parent, kp_child, controls_parent, 0, spot_man))
    p4 = Process(target=holo, args=(lock, spot_lock, trap_parent, kp_child, controls_parent, 1, spot_man)) # likely running into concurrency issues
    p2 = Process(target=simulator, args=(lock, spot_lock, trap_child, kp_parent, controls_child))
    #p5 = Process(target=sync_pipes, args=(controls_child_one, controls_child_two, controls_parent))
    p3 = Process(target=init_holo_engine, args=())

    p2.start()
    p1.start()
    p3.start()
    p4.start()
    #p0.start()
    #5.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    #p0.join()
    #p5.start()