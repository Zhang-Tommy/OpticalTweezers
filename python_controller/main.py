import multiprocessing
import random
import keyboard
import camera

from mpc import *
from sim_manager import SimManager
from simulator import white_bg
from spot_manager import SpotManager
from utilities import *
from multiprocessing import Lock, Process

def holo(controls_parent, target_bead, spot_man, goal, start=None, is_donut=False, is_line=False):
    kps = spot_man.get_obstacles() # Blocking, wait until keypoints have been received

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

    #print(f"Before {len(kpsarray)}")
    kpsarray = kpsarray[~jnp.all(kpsarray == start_state, axis=1)]

    current_length = kpsarray.shape[0]
    if current_length < KPS_SIZE:
        padding_length = KPS_SIZE - current_length
        pad_array = jnp.zeros((padding_length, kpsarray.shape[1]))
        kpsarray = jnp.vstack([kpsarray, pad_array])

    #print(kpsarray)
    #print(f"After {len(kpsarray)}")
    #print(f"KPS LEN CREATE: {len(kpsarray)}")
    env = Environment.create(len(kpsarray), kpsarray)

    init_control = gen_initial_traj(start_state, goal_position, N).T

    state = start_state
    dynamics = RK4Integrator(ContinuousTimeBeadDynamics(), DT)

    if is_donut:
        setattr(RunningCost, 'obstacle_separation', 1.4 * OBSTACLE_SEPARATION)
    elif is_line:
        setattr(RunningCost, 'obstacle_separation', 1.6 * OBSTACLE_SEPARATION)
    else:
        setattr(RunningCost, 'obstacle_separation', OBSTACLE_SEPARATION)

    while True:
        st = time.time()
        empty_env = Environment.create(0, jnp.array([]))


        solution = policy(state, goal_position, init_control, env, dynamics, RunningCost, MPCTerminalCost, empty_env,False, N)
        states, opt_controls = solution["optimal_trajectory"]
        control = opt_controls[0]
        try:
            spot_man.move_trap((int(state[0]), int(state[1])), (int(control[0]), int(control[1])))
        except:
            print(f"Trap move out of bounds invalid: {state[0]}, {state[1]} to {control[0]}, {control[1]}")
        init_control = opt_controls
        prev_state = state
        state = control  # The control is the position of the bead (wherever we place the trap is wherever the bead will go)

        controls_parent.send(opt_controls)
        #  Sets a range where obtsacles are visible, reduce to local knowledge of enivornment
        # nearest_kps = []
        # for i, kp in enumerate(kpsarray):
        #     if np.linalg.norm(np.array([state[0], state[1]]) - np.array(kp)) < 100:
        #         nearest_kps.append(kp)
        #
        # if len(nearest_kps) < 50:
        #     nearest_kps.extend([[0.0, 0.0]] * (50 - len(nearest_kps)))

        #nearest_kps = jnp.asarray(nearest_kps)
        #env = env.update(kpsarray, len(kpsarray))

        kpsarray = jnp.asarray(spot_man.get_obstacles())

        kpsarray = kpsarray[~jnp.all(kpsarray == np.array([int(control[0]), int(control[1])]), axis=1)]
        nearest_kps = []
        for kp in kpsarray:
            if np.linalg.norm(np.array([state[0], state[1]]) - np.array(kp)) < 100:
                nearest_kps.append(kp)


        kpsarray = jnp.asarray(nearest_kps)
        # nearest_kps = []
        # for i, kp in enumerate(kpsarray):
        #     if np.linalg.norm(np.array([state[0], state[1]]) - np.array(kp)) < 100:
        #         nearest_kps.append(kp)
        #
        # #if len(nearest_kps) < 50:
        # #    nearest_kps.extend([[0.0, 0.0]] * (50 - len(nearest_kps)))
        #
        # kpsarray = jnp.asarray(nearest_kps)

        current_length = kpsarray.shape[0]
        if current_length == 0:
            print("AHA")
        if current_length < KPS_SIZE:
            padding_length = KPS_SIZE - current_length
            pad_array = jnp.zeros((padding_length, kpsarray.shape[1]))
            kpsarray = jnp.vstack([kpsarray, pad_array])

        env = env.update(kpsarray, len(kpsarray))
        #print(f"KPS LEN UPDATE: {len(kpsarray)}")

        dist_to_goal = np.sqrt((state[0] - goal_position[0])**2 + (state[1] - goal_position[1])**2)
        if dist_to_goal < 2:
            print("Goal reached!")
            break

        if keyboard.is_pressed('q'):
            break
        et = time.time()

        # rudimentary timing controller
        if 1 / (et - st) > 20 and 0.05 - (et - st) > 0:
            time.sleep(0.05 - (et - st))


def simulator(spot_man, controls_child, controls_parent):
    """ Controls the simulator visualization with random bead distribution """
    sim_man = SimManager()
    number_of_beads = 25
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

    while True:
        ctrl_zero = cv2.getTrackbarPos('MoveToGoals', 'Optical Tweezers Simulator')
        ctrl_one = cv2.getTrackbarPos('MoveDonutLineToGoal', 'Optical Tweezers Simulator')
        if ctrl_zero:
            # for each goal position, find a obstacle bead and create controller process
            for i, goal in enumerate(spot_man.get_goal_pos().keys()):
                p = Process(target=holo,
                             args=(controls_parent, i, spot_man, goal))
                p.start()
            cv2.setTrackbarPos('MoveToGoals', 'Optical Tweezers Simulator', 0)
        elif ctrl_one:
            p0 = Process(target=holo,
                        args=(controls_parent, None, spot_man, donut_goal, donut_start, True, False))
            p0.start()

            p1 = Process(target=holo,
                        args=(controls_parent, None, spot_man, line_goal, line_start, False, True))
            p1.start()
            cv2.setTrackbarPos('MoveDonutLineToGoal', 'Optical Tweezers Simulator', 0)

        frame = white_bg.copy()
        sim_man.brownian_move(dt, dynamics)

        frame = draw_traps(spot_man, frame, sim_man, donut_goal, line_goal)

        # Handle controls and draw old/new controls
        if controls_child.poll():
            opt_controls = controls_child.recv().reshape(-1, 2)
            for g, cont in enumerate(opt_controls):
                if g % 5 == 0:
                    cv2.circle(frame, (int(cont[0]), int(cont[1])), 2, (128, 0, 0), -1)
                pass

        # Detect and draw beads using keypoints
        key_points = camera.detect_beads(frame, is_simulator=True)
        cv2.drawKeypoints(frame, key_points, frame, (255, 0, 0))

        points = [[kp.pt[0], kp.pt[1]] for kp in key_points]
        spot_man.set_obstacles(points)

        cv2.imshow("Optical Tweezers Simulator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            os.system("taskkill /f /im hologram_engine_64.exe")
            break

    cv2.destroyAllWindows()

def cam(spot_man, controls_child, controls_parent):
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

    while True:
        with ia.fetch() as buffer:
            component = buffer.payload.components[0]
            img = np.ndarray(buffer=component.data.copy(), dtype=np.uint8,
                             shape=(component.height, component.width, 1))

        key_points = camera.detect_beads(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ctrl_zero = cv2.getTrackbarPos('MoveToGoals', 'Optical Tweezers Simulator')
        ctrl_one = cv2.getTrackbarPos('MoveDonutLineToGoal', 'Optical Tweezers Simulator')
        if ctrl_zero:
            # for each goal position, find a obstacle bead and create controller process
            for i, goal in enumerate(spot_man.get_goal_pos().keys()):
                if i < len(spot_man.get_trapped_beads().keys()):
                    p = Process(target=holo,
                                args=(controls_parent, i, spot_man, goal))
                    p.start()
            cv2.setTrackbarPos('MoveToGoals', 'Optical Tweezers Simulator', 0)
        elif ctrl_one:
            p0 = Process(target=holo,
                         args=(controls_parent, None, spot_man, donut_goal, donut_start, True, False))
            p0.start()

            p1 = Process(target=holo,
                         args=(controls_parent, None, spot_man, line_goal, line_start, False, True))
            p1.start()
            cv2.setTrackbarPos('MoveDonutLineToGoal', 'Optical Tweezers Simulator', 0)

        img = draw_traps(spot_man, img, 2, donut_goal, line_goal)

        if controls_child.poll():
            opt_controls = controls_child.recv().reshape(-1, 2)
            for g, cont in enumerate(opt_controls):
                if g % 5 == 0:
                    cv2.circle(img, (int(cont[0]), int(cont[1])), 2, (128, 0, 0), -1)

        cv2.drawKeypoints(img, key_points, img, (255, 0, 0))
        cv2.imshow('Optical Tweezers Simulator', img)
        points = [[kp.pt[0], kp.pt[1]] for kp in key_points]
        spot_man.set_obstacles(points)
        cv2.waitKey(1)
        k += 1

        if keyboard.is_pressed('q'):
            ia.stop()
            ia.destroy()
            os.system("taskkill /f /im  hologram_engine_64.exe")
            break

if __name__ == "__main__":
    SpotManagerManager.register('SpotManager', SpotManager)
    SpotManagerManager.register('get_trapped_beads', SpotManager.get_trapped_beads)
    SpotManagerManager.register('add_spot', SpotManager.add_spot)
    SpotManagerManager.register('move_trap', SpotManager.move_trap)
    SpotManagerManager.register('remove_trap', SpotManager.remove_trap)

    manager = SpotManagerManager()
    manager.start()
    frame_queue = multiprocessing.Queue()  # Queue for camera frames
    controls_parent, controls_child = multiprocessing.Pipe() # Communicates future optimal controls to simulator

    spot_man = manager.SpotManager()
    lock = Lock()
    spot_lock = Lock()

    #p2 = Process(target=simulator, args=(spot_man, controls_child, controls_parent))
    p3 = Process(target=init_holo_engine, args=())

    p0 = Process(target=cam, args=(spot_man, controls_child, controls_parent))

    p0.start()
    p3.start()

    p0.join()
    p3.join()

