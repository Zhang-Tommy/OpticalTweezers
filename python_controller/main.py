import multiprocessing
import random
import threading
from multiprocessing import Lock, Process
from multiprocessing.managers import BaseManager

import cv2
import keyboard
import concurrent.futures
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

        lock.acquire()
        trap_parent.send(traps) # can just try with two, make non-scalable
        controls_parent.send(opt_controls)
        lock.release()
        if kp_child.poll():
            kps = kp_child.recv()
            kpsarray = jnp.asarray(kps)
            kpsarray = remove_closest_point(kpsarray, prev_state[0], prev_state[1])

            nearest_kps = []
            for i, kp in enumerate(kpsarray):
                if np.linalg.norm(np.array([state[0], state[1]]) - np.array(kp)) < 100:
                    nearest_kps.append(kp)

            if len(nearest_kps) < 50:
                nearest_kps.extend([[0.0, 0.0]] * (50 - len(nearest_kps)))

            nearest_kps = jnp.asarray(nearest_kps)
            env = env.update(nearest_kps, len(nearest_kps))

        ## if close to goal, break
        dist_to_goal = np.sqrt((state[0] - goal_position[0])**2 + (state[1] - goal_position[1])**2)
        if dist_to_goal < 5:
            print("Goal reached!")
            break

        if keyboard.is_pressed('q'):
            break
        et = time.time()

        # rudimentary timing controller
        if 1 / (et - st) > 20 and 0.05 - (et - st) > 0:
            time.sleep(0.05 - (et - st))

def simulator(lock, spot_lock, trap_child, kp_parent, controls_child, spot_man):
    """ Controls the simulator visualization with random bead distribution """
    sim_man = SimManager()
    number_of_beads = 50
    dt = 0.0015
    dragging_trap_idx = [None]

    # Initialize beads and traps
    for _ in range(number_of_beads):
        x_start = random.randint(0, CAM_X - 1)
        y_start = random.randint(0, CAM_Y - 1)
        sim_man.add_bead((x_start, y_start))

    traps = []

    def nothing(x):
        pass

    cv2.namedWindow("Optical Tweezers Simulator")
    cv2.setMouseCallback("Optical Tweezers Simulator", mouse_callback, param=(spot_man, traps, dragging_trap_idx))
    cv2.createTrackbar('R', 'Optical Tweezers Simulator', 0, 255, nothing)
    dynamics = ContinuousTimeObstacleDynamics()

    i = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            frame = white_bg.copy()

            sim_man.brownian_move(dt, dynamics)

            if trap_child.poll():  # Poll for updated trap data (non-blocking)
                traps = trap_child.recv()


            for x, y in traps:   # Draw traps
                cv2.circle(frame, (x, y), 10, (0, 128, 0), -1)

            # Handle controls and draw old/new controls
            if controls_child.poll():
                opt_controls = controls_child.recv().reshape(-1, 2)
                for g, cont in enumerate(opt_controls):
                    if g % 25 == 0:
                        cv2.circle(frame, (int(cont[0]), int(cont[1])), 2, (128, 0, 0), -1)

            # Detect and draw beads using keypoints
            key_points = camera.detect_beads(frame)
            cv2.drawKeypoints(frame, key_points, frame, (255, 0, 0))

            # Send keypoints asynchronously
            points = [[kp.pt[0], kp.pt[1]] for kp in key_points]
            # Todo: this will probably not work in experiments since beads can freely diffuse into the workspace
            if len(points) < number_of_beads:  # pad the keypoints with zeros to keep consistent length
                points.extend([[0.0, 0.0]] * (number_of_beads - len(points)))

            # Send keypoints
            executor.submit(kp_parent.send, points)

            cv2.imshow("Optical Tweezers Simulator", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                os.system("taskkill /f /im hologram_engine_64.exe")
                break

    cv2.destroyAllWindows()

def mouse_callback(event, x, y, flags, param):
    spot_man, traps, dragging_trap_idx = param

    if event == cv2.EVENT_LBUTTONDOWN:  # Left click to add or select trap
        for i, trap in enumerate(traps):
            if np.linalg.norm(np.array([x, y]) - np.array(trap)) < 15:
                dragging_trap_idx[0] = i  # Start dragging this trap
                return
        # Add new trap if none selected
        spot_man.add_spot((x, y))
        traps.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:  # Right click to remove trap
        for i, trap in enumerate(traps):
            if np.linalg.norm(np.array([x, y]) - np.array(trap)) < 15:
                spot_man.remove_trap((trap[0], trap[1]))
                traps.pop(i)
                return

    elif event == cv2.EVENT_MOUSEMOVE:  # Dragging trap around
        if dragging_trap_idx[0] is not None:
            spot_man.move_trap((traps[dragging_trap_idx[0]][0], traps[dragging_trap_idx[0]][1]), (x, y))
            traps[dragging_trap_idx[0]] = (x, y)
            #print(traps)
            return

    elif event == cv2.EVENT_LBUTTONUP:  # Release dragging
        dragging_trap_idx[0] = None

    elif event == cv2.EVENT_LBUTTONDBLCLK:
        # Todo: Select a bead? turn it a different color, then slider can do things to it, switch to line trap, etc...
        pass
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        # Todo: deselect a bead
        pass

def cam(kp_parent, trap_child, spot_man, controls_child):
    #vid = cv2.VideoCapture(r'.\testing_video1.mp4')
    ia = start_image_acquisition()
    dragging_trap_idx = [None]
    k = 0
    def nothing(x):
        pass

    cv2.namedWindow("Camera Feed")
    traps = []
    points = []
    points.extend([[0.0, 0.0]] * 100)
    cv2.setMouseCallback("Camera Feed", mouse_callback, param=(spot_man, traps, dragging_trap_idx))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        #while vid.isOpened():
        while True:
            with ia.fetch() as buffer:
                component = buffer.payload.components[0]
                img = np.ndarray(buffer=component.data.copy(), dtype=np.uint8,
                                 shape=(component.height, component.width, 1))

            #ret, img = vid.read()
            #if ret:
            key_points = camera.detect_beads(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #if trap_child.poll():
            #    traps = trap_child.recv()

            traps = list(spot_man.get_trapped_beads().keys())
            for x, y in traps:  # Draw traps
                cv2.circle(img, (x, y), 15, (256, 0, 256), 1)

            if controls_child.poll():
                opt_controls = controls_child.recv().reshape(-1, 2)
                for g, cont in enumerate(opt_controls):
                    if g % 5 == 0:
                        cv2.circle(img, (int(cont[0]), int(cont[1])), 2, (128, 0, 0), -1)

            points = [[kp.pt[0], kp.pt[1]] for kp in key_points]
            if len(points) < 100:  # pad the keypoints with zeros to keep consistent length
                points.extend([[0.0, 0.0]] * (100 - len(points)))

            if k % 5 == 0:
                executor.submit(kp_parent.send, points)
            cv2.drawKeypoints(img, key_points, img, (255, 0, 0))

            cv2.waitKey(1)
            cv2.imshow('Camera Feed', img)
            k += 1

            if keyboard.is_pressed('q'):
                #ia.stop()
                #ia.destroy()
                os.system("taskkill /f /im  hologram_engine_64.exe")
                break

class SpotManagerManager(BaseManager):
    pass

if __name__ == "__main__":
    # Todo: Build a higher-level controller!
    SpotManagerManager.register('SpotManager', SpotManager)
    SpotManagerManager.register('get_trapped_beads', SpotManager.get_trapped_beads)
    SpotManagerManager.register('add_spot', SpotManager.add_spot)
    SpotManagerManager.register('move_trap', SpotManager.move_trap)
    SpotManagerManager.register('move_trap', SpotManager.remove_trap)

    manager = SpotManagerManager()
    manager.start()
    frame_queue = multiprocessing.Queue()  # Queue for camera frames
    kp_parent, kp_child = multiprocessing.Pipe()  # Communicates keypoints between processes
    trap_parent, trap_child = multiprocessing.Pipe()  # Communicates trap positions between processes
    controls_parent, controls_child = multiprocessing.Pipe() # Communicates future optimal controls to simulator

    spot_man = manager.SpotManager()
    lock = Lock()
    spot_lock = Lock()

    p1 = Process(target=holo, args=(lock, spot_lock, trap_parent, kp_child, controls_parent, 0, spot_man))
    #p4 = Process(target=holo, args=(lock, spot_lock, trap_parent, kp_child, controls_parent, 1, spot_man)) # likely running into concurrency issues
    #p2 = Process(target=simulator, args=(lock, spot_lock, trap_child, kp_parent, controls_child, spot_man))
    p3 = Process(target=init_holo_engine, args=())
    p0 = Process(target=cam, args=(kp_parent, trap_child, spot_man, controls_child))

    #p2.start()
    p1.start()
    p3.start()
    #p4.start()
    p0.start()

    p1.join()
    #p2.join()
    p3.join()
    #p4.join()
    p0.join()
