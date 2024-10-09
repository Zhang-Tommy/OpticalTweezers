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

# Todo: Ensure the controller is aware of beads that other controllers are moving
def holo(kp_child, controls_parent, target_bead, spot_man, goal, start=None, is_donut=False, is_line=False):
    kps = kp_child.recv() # Blocking, wait until keypoints have been received

    if not is_donut and not is_line:
        # Choose first detected bead as our target bead
        x_start = float(kps[target_bead][0])
        y_start = float(kps[target_bead][1])
        # Trap the target bead
        spot_man.add_spot((int(x_start), int(y_start)))
        goal_position = jnp.array([goal[0], goal[1]])
    elif is_donut:
        start_pos = start[0]
        spot_man.add_spot((start_pos[0], start_pos[1]), is_donut=True)
        x_start = float(start_pos[0])
        y_start = float(start_pos[1])
        goal_position = jnp.array([goal[0][0], goal[0][1]])
    elif is_line:
        start_pos = start[0]
        spot_man.add_spot((start_pos[0], start_pos[1]), is_line=True)
        x_start = float(start_pos[0])
        y_start = float(start_pos[1])
        goal_position = jnp.array([goal[0][0], goal[0][1]])

    # Define start and goal states
    start_state = jnp.array([x_start, y_start])

    kpsarray = jnp.asarray(kps)
    kpsarray = kpsarray[kpsarray != start_state]

    env = Environment.create(int(len(kpsarray)/2), kpsarray)

    init_control = gen_initial_traj(start_state, goal_position, N).T

    state = start_state
    dynamics = RK4Integrator(ContinuousTimeBeadDynamics(), DT)

    while True:
        try:
            st = time.time()
            empty_env = Environment.create(0, jnp.array([]))
            solution = policy(state, goal_position, init_control, env, dynamics, RunningCost, MPCTerminalCost, empty_env, False, N)
            states, opt_controls = solution["optimal_trajectory"]
            control = opt_controls[0]
            try:
                spot_man.move_trap((int(state[0]), int(state[1])), (int(control[0]), int(control[1])))
            except:
                print(f"Trap move out of bounds invalid: {state[0]}, {state[1]} to {control[0]}, {control[1]}")

            prev_state = state
            state = control  # The control is the position of the bead (wherever we place the trap is wherever the bead will go)

            controls_parent.send(opt_controls)
            if kp_child.poll():
               # kps = spot_man.get_obstacles()
                kps = kp_child.recv()
                kpsarray = jnp.asarray(kps)
                kpsarray = remove_closest_point(kpsarray, prev_state[0], prev_state[1])


                #  Sets a range where obtsacles are visible, reduce to local knowledge of enivornment
                nearest_kps = []
                for i, kp in enumerate(kpsarray):
                    if np.linalg.norm(np.array([state[0], state[1]]) - np.array(kp)) < 100:
                        nearest_kps.append(kp)

                if len(nearest_kps) < 50:
                    nearest_kps.extend([[0.0, 0.0]] * (50 - len(nearest_kps)))

                nearest_kps = jnp.asarray(nearest_kps)
                env = env.update(nearest_kps, len(nearest_kps))

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
        except Exception as e: print(e)

def simulator(kp_parent, controls_child, spot_man, lock, spot_lock, trap_parent, kp_child, controls_parent):
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

    traps, donut_start, donut_goal, line_start, line_goal = [], [], [], [], []

    def nothing(x):
        pass

    cv2.namedWindow("Optical Tweezers Simulator")
    params = [spot_man, traps, dragging_trap_idx, donut_start, donut_goal, line_start, line_goal]
    cv2.setMouseCallback("Optical Tweezers Simulator", mouse_callback, param=params)
    cv2.createTrackbar('LineTrap', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('DonutTrap', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('MoveToGoals', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('MoveDonutLineToGoal', 'Optical Tweezers Simulator', 0, 1, nothing)

    dynamics = ContinuousTimeObstacleDynamics()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            ctrl_zero = cv2.getTrackbarPos('MoveToGoals', 'Optical Tweezers Simulator')
            ctrl_one = cv2.getTrackbarPos('MoveDonutLineToGoal', 'Optical Tweezers Simulator')
            if ctrl_zero:
                # for each goal position, find a obstacle bead and create controller process
                for i, goal in enumerate(spot_man.get_goal_pos().keys()):
                    p = Process(target=holo,
                                 args=(kp_child, controls_parent, i, spot_man, goal))
                    p.start()
                cv2.setTrackbarPos('MoveToGoals', 'Optical Tweezers Simulator', 0)
            elif ctrl_one:
                # # start controllers with annular trap and line trap\
                print(donut_start)
                print(line_start)
                p0 = Process(target=holo,
                            args=(kp_child, controls_parent, None, spot_man, donut_goal, donut_start, True, False))
                p0.start()

                p1 = Process(target=holo,
                            args=(kp_child, controls_parent, None, spot_man, line_goal, line_start, False, True))
                p1.start()
                cv2.setTrackbarPos('MoveDonutLineToGoal', 'Optical Tweezers Simulator', 0)

            frame = white_bg.copy()
            sim_man.brownian_move(dt, dynamics)

            frame = draw_traps(spot_man, frame, sim_man)

            # Handle controls and draw old/new controls
            if controls_child.poll():
                opt_controls = controls_child.recv().reshape(-1, 2)
                for g, cont in enumerate(opt_controls):
                    if g % 25 == 0:
                        cv2.circle(frame, (int(cont[0]), int(cont[1])), 2, (128, 0, 0), -1)

            # Detect and draw beads using keypoints
            key_points = camera.detect_beads(frame, is_simulator=True)
            cv2.drawKeypoints(frame, key_points, frame, (255, 0, 0))

            kp_set = {tuple(map(int, kp.pt)) for kp in key_points}

            #if len(kp_set) < number_of_beads:

            spot_man.set_obstacles(kp_set)

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
    spot_man, traps, dragging_trap_idx, donut_start, donut_goal, line_start, line_goal = param
    line_trap = cv2.getTrackbarPos('LineTrap', 'Optical Tweezers Simulator')
    donut_trap = cv2.getTrackbarPos('DonutTrap', 'Optical Tweezers Simulator')
    if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_ALTKEY: # donut
        donut_goal.append((x, y))
        print("Donut Goal Added")
    elif event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY: # line
        line_goal.append((x, y))
        print("Line Goal Added")
    elif event == cv2.EVENT_LBUTTONDOWN:  # Left click to add or select trap
        for i, trap in enumerate(traps):
            if np.linalg.norm(np.array([x, y]) - np.array(trap)) < 15:
                dragging_trap_idx[0] = i  # Start dragging this trap
                return
        # Add new trap if none selected
        if line_trap:
            spot_man.add_spot((x,y), is_line=True)
            line_start.append((x, y))
        elif donut_trap:
            spot_man.add_spot((x,y), is_donut=True)
            donut_start.append((x, y))
        else:
            spot_man.add_spot((x,y))

        traps.append((x,y))
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right click to remove trap
        for j, trap in enumerate(traps):
            if np.linalg.norm(np.array([x,y]) - np.array(trap)) < 15:
                spot_man.remove_trap((trap[0], trap[1]))
                traps.pop(j)
                return
        for goal in spot_man.get_goal_pos().keys():
            if np.linalg.norm(np.array([x,y]) - np.array(goal)) < 15:
                spot_man.remove_goal_pos((goal[0], goal[1]))
                return
    elif event == cv2.EVENT_MOUSEMOVE:  # Dragging trap around
        if dragging_trap_idx[0] is not None:
            spot_man.move_trap((traps[dragging_trap_idx[0]][0], traps[dragging_trap_idx[0]][1]), (x, y))
            traps[dragging_trap_idx[0]] = (x, y)
            #print(traps)
            return
    elif event == cv2.EVENT_LBUTTONUP:  # Release dragging
        dragging_trap_idx[0] = None
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        # Todo: deselect a bead
        pass
    elif event == cv2.EVENT_MBUTTONDOWN:
        # add goal points
        spot_man.add_goal_pos((x,y))

def cam(kp_parent, trap_child, spot_man, controls_child):
    #vid = cv2.VideoCapture(r'.\testing_video1.mp4')
    ia = start_image_acquisition()
    dragging_trap_idx = [None]
    k = 0
    def nothing(x):
        pass

    traps, points, donut_start, donut_goal, line_start, line_goal = [], [], [], [], [], []
    points.extend([[0.0, 0.0]] * 100)

    cv2.namedWindow("Optical Tweezers Simulator")
    params = [spot_man, traps, dragging_trap_idx, donut_start, donut_goal, line_start, line_goal]
    cv2.setMouseCallback("Optical Tweezers Simulator", mouse_callback, param=params)
    cv2.createTrackbar('LineTrap', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('DonutTrap', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('MoveToGoals', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('MoveDonutLineToGoal', 'Optical Tweezers Simulator', 0, 1, nothing)
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

            ctrl_zero = cv2.getTrackbarPos('MoveToGoals', 'Optical Tweezers Simulator')
            ctrl_one = cv2.getTrackbarPos('MoveDonutLineToGoal', 'Optical Tweezers Simulator')
            if ctrl_zero:
                # for each goal position, find a obstacle bead and create controller process
                for i, goal in enumerate(spot_man.get_goal_pos().keys()):
                    p = Process(target=holo,
                                args=(kp_child, controls_parent, i, spot_man, goal))
                    p.start()
                cv2.setTrackbarPos('MoveToGoals', 'Optical Tweezers Simulator', 0)
            elif ctrl_one:
                # # start controllers with annular trap and line trap\
                print(donut_start)
                print(line_start)
                p0 = Process(target=holo,
                             args=(kp_child, controls_parent, None, spot_man, donut_goal, donut_start, True, False))
                p0.start()

                p1 = Process(target=holo,
                             args=(kp_child, controls_parent, None, spot_man, line_goal, line_start, False, True))
                p1.start()
                cv2.setTrackbarPos('MoveDonutLineToGoal', 'Optical Tweezers Simulator', 0)

            traps = spot_man.get_trapped_beads()

            img = draw_traps(traps, img)

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
    SpotManagerManager.register('SpotManager', SpotManager)
    SpotManagerManager.register('get_trapped_beads', SpotManager.get_trapped_beads)
    SpotManagerManager.register('add_spot', SpotManager.add_spot)
    SpotManagerManager.register('move_trap', SpotManager.move_trap)
    SpotManagerManager.register('remove_trap', SpotManager.remove_trap)

    manager = SpotManagerManager()
    manager.start()
    frame_queue = multiprocessing.Queue()  # Queue for camera frames
    kp_parent, kp_child = multiprocessing.Pipe()  # Communicates keypoints between processes
    trap_parent, trap_child = multiprocessing.Pipe()  # Communicates trap positions between processes
    controls_parent, controls_child = multiprocessing.Pipe() # Communicates future optimal controls to simulator

    spot_man = manager.SpotManager()
    lock = Lock()
    spot_lock = Lock()

    p2 = Process(target=simulator, args=(kp_parent, controls_child, spot_man, lock, spot_lock, trap_parent, kp_child, controls_parent))
    p3 = Process(target=init_holo_engine, args=())
    #p0 = Process(target=cam, args=(kp_parent, trap_child, spot_man, controls_child))

    p2.start()
    p3.start()

    p2.join()
    p3.join()

