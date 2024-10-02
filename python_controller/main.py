import multiprocessing
import random
import threading
from multiprocessing import Lock, Process
from multiprocessing.managers import BaseManager

import cv2
import keyboard

import camera
from mpc import *
from sim_manager import SimManager
from simulator import white_bg
from spot_manager import SpotManager
from utilities import *

def holo(lock, spot_lock, trap_parent, kp_child, controls_parent, target_bead, spot_man):
    spot_lock.acquire()
    lock.acquire()
    kps = kp_child.recv() # Blocking, wait until keypoints have been received

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

    kpsarray = jnp.asarray(kps)
    kpsarray = kpsarray[kpsarray != start_state]  # remove start state from keypoints
    #psarray = kpsarray[kpsarray != start_state[1]]

    env = Environment.create(num_beads, kpsarray) ## problematic

    print(f"KPS: {len(kpsarray)}, NUM_BEADS: {num_beads}")

    N = 125
    init_control = gen_initial_traj(start_state, goal_position, N).T

    state = start_state

    dynamics = RK4Integrator(ContinuousTimeBeadDynamics(), DT)

    lock.release()

    while True:
        st = time.time()
        empty_env = Environment.create(0, jnp.array([]))
        solution = policy(state, goal_position, init_control, env, dynamics, RunningCost, MPCTerminalCost, empty_env, False, N)
        states, opt_controls = solution["optimal_trajectory"]
        control = opt_controls[0]
        #spot_lock.acquire()
        try:
            spot_man.move_trap((int(state[0]), int(state[1])), (int(control[0]), int(control[1])))
        except:
            print(f"Trap move out of bounds invalid: {state[0]}, {state[1]} to {control[0]}, {control[1]}")

        traps = list(spot_man.get_trapped_beads().keys())
        #spot_lock.release()

        prev_state = state
        state = control  # The control is the position of the bead (wherever we place the trap is wherever the bead will go)

        #lock.acquire()
        trap_parent.send(traps) # can just try with two, make non-scalable
        controls_parent.send(opt_controls)
        #lock.release()
        if kp_child.poll():
            kps = kp_child.recv()
            kpsarray = jnp.asarray(kps)
            num_beads = len(kps)
            # we need to ensure the bead we are moving is removed from this
            kpsarray = remove_closest_point(kpsarray, prev_state[0], prev_state[1])
            #kpsarray = kpsarray[1:]

            env = env.update(kpsarray, num_beads-1)

        time.sleep(.015)

        ## if close to goal, break
        dist_to_goal = np.sqrt((state[0] - goal_position[0])**2 + (state[1] - goal_position[1])**2)
        if dist_to_goal < 5:
            #end_time = time.time()
            #elapsed_time = end_time - start_time
            #print(f'Goal Reached in {elapsed_time} seconds!')
            break

        if keyboard.is_pressed('q'):
            break
        et = time.time()

        # rudimentary timing controller
        if 1 / (et - st) > 20 and 0.05 - (et - st) > 0:
            time.sleep(0.05 - (et - st))

def simulator(lock, spot_lock, trap_child, kp_parent, controls_child):
    """ Controls the simulator visualization w/random bead distribution """
    # runs at about 12.5 hz
    sim_man = SimManager()
    number_of_beads = 50

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
            for t, cont in enumerate(old_controls):
                if t % 5 == 0:
                    cv2.circle(white_bg, (int(cont[0]), int(cont[1])), 2, (255, 255, 255), -1)

            opt_controls = controls_child.recv()
            opt_controls = opt_controls.reshape(-1, 2)
            old_controls = opt_controls

            for g, cont in enumerate(opt_controls):
                if g % 5 == 0:
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

    p1 = Process(target=holo, args=(lock, spot_lock, trap_parent, kp_child, controls_parent, 0, spot_man))
    p4 = Process(target=holo, args=(lock, spot_lock, trap_parent, kp_child, controls_parent, 1, spot_man)) # likely running into concurrency issues
    p2 = Process(target=simulator, args=(lock, spot_lock, trap_child, kp_parent, controls_child))
    p3 = Process(target=init_holo_engine, args=())

    p2.start()
    p1.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
