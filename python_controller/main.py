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
from sklearn.cluster import DBSCAN

def holo(controls_parent, spot_man, goal, start, is_donut=False, is_line=False):
    # Get the first view of obstacles
    kps = spot_man.get_obstacles()

    if is_donut:
        start_pos = start
        spot_man.add_spot((start_pos[0], start_pos[1]), is_donut=True)
        x_start = float(start_pos[0])
        y_start = float(start_pos[1])
        goal_position = jnp.array([goal[0], goal[1]])
        N_n = N #100
        setattr(RunningCost, 'obstacle_separation', 1.2 * OBSTACLE_SEPARATION)
    elif is_line:
        start_pos = start
        spot_man.add_spot((start_pos[0], start_pos[1]), is_line=True)
        x_start = float(start_pos[0])
        y_start = float(start_pos[1])
        goal_position = jnp.array([goal[0], goal[1]])
        N_n = N # 140
        setattr(RunningCost, 'obstacle_separation', 1 * OBSTACLE_SEPARATION)
        setattr(RunningCost, 'agent_as_ellipse', True)
        setattr(RunningCost, 'ellipse_axis', [ELLIPSE_MAJOR_AXIS, ELLIPSE_MINOR_AXIS])
    else:
        # Choose first detected bead as our target bead
        x_start = float(start[0])
        y_start = float(start[1])
        # Trap the target bead
        #spot_man.add_spot((int(x_start), int(y_start)))
        goal_position = jnp.array([goal[0], goal[1]])
        N_n = N
        setattr(RunningCost, 'obstacle_separation', OBSTACLE_SEPARATION)

    start_state = jnp.array([x_start, y_start])

    kpsarray = jnp.asarray(kps)
    kpsarray = kpsarray[~jnp.all(kpsarray == start_state, axis=1)] # remove start state from obstacles

    current_length = kpsarray.shape[0]

    if current_length < KPS_SIZE:  # pad obstacles array so jax doesn't recompile
        padding_length = KPS_SIZE - current_length
        pad_array = jnp.zeros((padding_length, kpsarray.shape[1]))
        kpsarray = jnp.vstack([kpsarray, pad_array])

    env = Environment.create(len(kpsarray), kpsarray)

    init_control = gen_initial_traj(start_state, goal_position, N_n).T  # generate initial guess for mpc

    state = start_state
    dynamics = RK4Integrator(ContinuousTimeBeadDynamics(), DT)
    time_step = 0
    ctrl_idx = 0
    empty_env = Environment.create(0, jnp.array([]))
    # We can decrease N as we get closer to the goal
    # But we shouldn't jax.jit recompile in the while loop
    # for i in range(35, N):
    #     init_control = gen_initial_traj(start_state, goal_position, i).T
    #     #policy(state, goal_position, init_control, env, dynamics, RunningCost, MPCTerminalCost, empty_env, False, i)
    #     lowered = jax.jit(policy, static_argnums=(4, 5, 6, 8, 9)).lower(state, goal_position, init_control, env, dynamics, RunningCost, MPCTerminalCost, empty_env, False, i)
    #     compiled = lowered.compile()
    #     print(f"Compiled {i}")

    while True:
        st = time.time()

        if time_step % MPC_COMPUTE_FREQ == 0:
            solution = policy(state, goal_position, init_control, env, dynamics, RunningCost, MPCTerminalCost, empty_env, False, N_n)
            states, opt_controls = solution["optimal_trajectory"]

        if time_step % MPC_COMPUTE_FREQ == 0:
            ctrl_idx = 0
        else:
            ctrl_idx += 1

        control = opt_controls[ctrl_idx]

        try:
            spot_man.move_trap((int(state[0]), int(state[1])), (int(control[0]), int(control[1])))
            #print(f"{state} -> {control}")
        except:
            print(f"Trap move out of bounds invalid: {state[0]}, {state[1]} to {control[0]}, {control[1]}")
            return


        init_control = opt_controls
        #init_control = gen_perturbed_controls(opt_controls, 10, 20, key)
        state = control  # The control is the position of the bead (wherever we place the trap is wherever the bead will go)

        controls_parent.send(opt_controls)

        kpsarray = jnp.asarray(spot_man.get_obstacles())

        kpsarray = kpsarray[~jnp.all(kpsarray == np.array([int(control[0]), int(control[1])]), axis=1)]
        nearest_kps = []

        kpsarray = remove_trapped_beads(state, kpsarray, is_donut, is_line)
        # if close to goal, remove obstacles that are close to goal as well

        dist_to_goal = np.sqrt((state[0] - goal_position[0]) ** 2 + (state[1] - goal_position[1]) ** 2)

        if dist_to_goal < GOAL_DIST_OBSTACLE_FREE and kpsarray.size != 0:
            distances = np.linalg.norm(kpsarray - goal_position, axis=1)
            kpsarray = kpsarray[distances > 75]

        # N_n = int(0.063 * dist_to_goal + 34.6)

        current_length = kpsarray.shape[0]
        if current_length < KPS_SIZE:
            padding_length = KPS_SIZE - current_length
            if kpsarray.size == 0:
                kpsarray = jnp.zeros((padding_length, 2))

            else:
                pad_array = jnp.zeros((padding_length, kpsarray.shape[1]))
                kpsarray = jnp.vstack([kpsarray, pad_array])

        env = env.update(kpsarray, len(kpsarray), time_step)

        if dist_to_goal < 2:
            print("Goal reached!")
            break

        if keyboard.is_pressed('q'):
            break
        et = time.time()
        time_step += 1
        # rudimentary timing controller
        print(et - st)
        if 1 / (et - st) > 20 and 0.05 - (et - st) > 0:
            time.sleep(0.05 - (et - st))

def remove_trapped_beads(current_state, kps_array, is_donut=False, is_line=False):
    filtered_kps = []

    if is_donut:
        # Donut trap: exclude beads within radius r_donut from current_state
        for kp in kps_array:
            distance = np.linalg.norm(current_state - kp)
            if distance > R_DONUT:
                filtered_kps.append(kp)

    elif is_line:
        # Line trap: exclude beads within a bounding box (width x length) centered around current_state
        half_width = L_WIDTH / 2.0
        half_length = L_LENGTH / 2.0

        for kp in kps_array:
            dx, dy = kp[0] - current_state[0], kp[1] - current_state[1]
            # Check if kp is outside the bounding box centered at current_state
            if abs(dx) > half_length or abs(dy) > half_width:
                filtered_kps.append(kp)

    else:
        # Point trap (default): exclude beads within radius r_point from current_state
        for kp in kps_array:
            distance = np.linalg.norm(current_state - kp)
            if distance > R_POINT:
                filtered_kps.append(kp)

    return np.array(filtered_kps)

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

    params = [spot_man, traps, dragging_trap_idx]

    def nothing(x):
        pass

    cv2.namedWindow("Optical Tweezers Simulator")
    cv2.setMouseCallback("Optical Tweezers Simulator", mouse_callback, param=params)
    cv2.createTrackbar('LineTrap', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('DonutTrap', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('PointGoal', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('MoveDonutLineToGoal', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('ClearObs', 'Optical Tweezers Simulator', 0, 1, nothing)

    dynamics = ContinuousTimeObstacleDynamics()

    while True:
        ctrl_zero = cv2.getTrackbarPos('PointGoal', 'Optical Tweezers Simulator')
        ctrl_one = cv2.getTrackbarPos('MoveDonutLineToGoal', 'Optical Tweezers Simulator')
        ctrl_two = cv2.getTrackbarPos('ClearObs', 'Optical Tweezers Simulator')
        line_trap = cv2.getTrackbarPos('LineTrap', 'Optical Tweezers Simulator')
        donut_trap = cv2.getTrackbarPos('DonutTrap', 'Optical Tweezers Simulator')

        if line_trap and donut_trap:
            cv2.setTrackbarPos('LineTrap', 'Optical Tweezers Simulator', 0)

        if ctrl_zero:
            point_starts = spot_man.get_start_pos()
            point_goals = spot_man.get_goal_pos()
            for start_pos in point_starts:
                if len(point_goals) != 0:
                    goal_pos = point_goals.popleft()
                    p = Process(target=holo,
                                args=(controls_parent, spot_man, goal_pos, start_pos))
                    p.start()
            #spot_man.clear_goals()
            spot_man.clear_starts()
            cv2.setTrackbarPos('PointGoal', 'Optical Tweezers Simulator', 0)
        elif ctrl_one:
            """In sequence of addition, start mpc towards goals from start positions for line and donut traps"""
            donut_starts = spot_man.get_start_pos(is_donut=True)
            line_starts = spot_man.get_start_pos(is_line=True)
            donut_goals = spot_man.get_goal_pos(is_donut=True)
            line_goals = spot_man.get_goal_pos(is_line=True)

            for start_pos in donut_starts:
                goal_pos = donut_goals.popleft()
                p = Process(target=holo,
                            args=(controls_parent, spot_man, goal_pos, start_pos, True, False))
                p.start()

            for start_pos in line_starts:
                goal_pos = line_goals.popleft()
                p = Process(target=holo,
                            args=(controls_parent, spot_man, goal_pos, start_pos, False, True))
                p.start()
            #spot_man.clear_goals(is_donut=True)
            spot_man.clear_starts(is_donut=True)
            #spot_man.clear_goals(is_line=True)
            spot_man.clear_starts(is_line=True)
            cv2.setTrackbarPos('MoveDonutLineToGoal', 'Optical Tweezers Simulator', 0)
        elif ctrl_two:
            if not spot_man.get_clearing_region():
                p = Process(target=clear_region,
                            args=(controls_parent, spot_man))
                p.start()
            cv2.setTrackbarPos('ClearObs', 'Optical Tweezers Simulator', 0)

        frame = white_bg.copy()
        sim_man.brownian_move(dt, dynamics)

        frame = draw_traps(spot_man, frame)

        # Handle controls and draw old/new controls
        if controls_child.poll():
            opt_controls = controls_child.recv().reshape(-1, 2)
            for g, cont in enumerate(opt_controls):
                if g % 5 == 0 and DEBUG:
                    cv2.circle(frame, (int(cont[0]), int(cont[1])), 2, (128, 0, 0), -1)

        # Detect and draw beads using keypoints
        key_points = camera.detect_beads(frame, is_simulator=True)
        if DEBUG:
            cv2.drawKeypoints(frame, key_points, frame, (255, 0, 0))

        points = [[kp.pt[0], kp.pt[1]] for kp in key_points]

        kps_array = np.asarray(points)

        clustering = DBSCAN(eps=60, min_samples=2, n_jobs=-1).fit(kps_array)
        frame, artifical_pts = create_artificial_obs(clustering, kps_array, frame)

        #cv2.rectangle(frame, (160,160), (480,320), (128, 0, 0), 1)

        spot_man.set_obstacles(points)
        spot_man.set_fake_obstacles(artifical_pts)

        cv2.imshow("Optical Tweezers Simulator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            os.system("taskkill /f /im hologram_engine_64.exe")
            break

    cv2.destroyAllWindows()

def cam(spot_man, controls_child, controls_parent):
    ia = start_image_acquisition()
    dragging_trap_idx = [None]
    k = 0

    traps, points, donut_start, donut_goal, line_start, line_goal = [], [], [], [], [], []
    points.extend([[0.0, 0.0]] * 100)

    params = [spot_man, traps, dragging_trap_idx]

    def nothing(x):
        pass

    cv2.namedWindow("Optical Tweezers Simulator")
    cv2.setMouseCallback("Optical Tweezers Simulator", mouse_callback, param=params)
    cv2.createTrackbar('LineTrap', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('DonutTrap', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('PointGoal', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('MoveDonutLineToGoal', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('ClearObs', 'Optical Tweezers Simulator', 0, 1, nothing)

    while True:
        with ia.fetch() as buffer:
            component = buffer.payload.components[0]
            img = np.ndarray(buffer=component.data.copy(), dtype=np.uint8,
                             shape=(component.height, component.width, 1))

        key_points = camera.detect_beads(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #donut_goal, donut_start, line_goal, line_start, spot_man, controls_parent = params
        ctrl_zero = cv2.getTrackbarPos('PointGoal', 'Optical Tweezers Simulator')
        ctrl_one = cv2.getTrackbarPos('MoveDonutLineToGoal', 'Optical Tweezers Simulator')
        ctrl_two = cv2.getTrackbarPos('ClearObs', 'Optical Tweezers Simulator')
        line_trap = cv2.getTrackbarPos('LineTrap', 'Optical Tweezers Simulator')
        donut_trap = cv2.getTrackbarPos('DonutTrap', 'Optical Tweezers Simulator')
        if line_trap and donut_trap:
            cv2.setTrackbarPos('LineTrap', 'Optical Tweezers Simulator', 0)
        if ctrl_zero:
            point_starts = spot_man.get_start_pos()
            point_goals = spot_man.get_goal_pos()
            for start_pos in point_starts:
                if len(point_goals) != 0:
                    goal_pos = point_goals.popleft()
                    p = Process(target=holo,
                                args=(controls_parent, spot_man, goal_pos, start_pos))
                    p.start()
            #spot_man.clear_goals()
            spot_man.clear_starts()
            cv2.setTrackbarPos('PointGoal', 'Optical Tweezers Simulator', 0)
        elif ctrl_one:
            """In sequence of addition, start mpc towards goals from start positions for line and donut traps"""
            donut_starts = spot_man.get_start_pos(is_donut=True)
            line_starts = spot_man.get_start_pos(is_line=True)
            donut_goals = spot_man.get_goal_pos(is_donut=True)
            line_goals = spot_man.get_goal_pos(is_line=True)

            for start_pos in donut_starts:
                goal_pos = donut_goals.popleft()
                p = Process(target=holo,
                            args=(controls_parent, spot_man, goal_pos, start_pos, True, False))
                p.start()

            for start_pos in line_starts:
                goal_pos = line_goals.popleft()
                p = Process(target=holo,
                            args=(controls_parent, spot_man, goal_pos, start_pos, False, True))
                p.start()
            #spot_man.clear_goals(is_donut=True)
            spot_man.clear_starts(is_donut=True)
            #spot_man.clear_goals(is_line=True)
            spot_man.clear_starts(is_line=True)
            cv2.setTrackbarPos('MoveDonutLineToGoal', 'Optical Tweezers Simulator', 0)
        elif ctrl_two:
            if not spot_man.get_clearing_region():
                p = Process(target=clear_region,
                            args=(controls_parent, spot_man))
                p.start()
            cv2.setTrackbarPos('ClearObs', 'Optical Tweezers Simulator', 0)

        img = draw_traps(spot_man, img)

        if controls_child.poll():
            opt_controls = controls_child.recv().reshape(-1, 2)
            # for g, cont in enumerate(opt_controls):
            #     if g % 1 == 0 and DEBUG:
            #         cv2.circle(img, (int(cont[0]), int(cont[1])), 2, (128, 0, 0), -1)

        if DEBUG:
            cv2.drawKeypoints(img, key_points, img, (255, 0, 0))
        cv2.imshow('Optical Tweezers Simulator', img)
        points = [[kp.pt[0], kp.pt[1]] for kp in key_points]

        kps_array = np.asarray(points)

        # if len(kps_array) > 2:
        #     clustering = DBSCAN(eps=60, min_samples=2).fit(kps_array)
        #     img, artifical_pts = create_artificial_obs(clustering, kps_array, img)
        #
        #     combined_pts = points + artifical_pts
        # else:
        combined_pts = points

        spot_man.set_obstacles(combined_pts)

        cv2.waitKey(1)
        k += 1

        if keyboard.is_pressed('q'):
            ia.stop()
            ia.destroy()
            os.system("taskkill /f /im  hologram_engine_64.exe")
            break

def clear_region(controls_parent, spot_man):
    """
    Using the locations of detected obstacles, actively control where obstacles are to create a clear workspace for experiments
    """
    # region defined by rectangle with corner pts at (160,160), (480,320)
    obs = spot_man.get_obstacles()
    spot_man.set_clearing_region(True)
    x_min, x_max = 160, 480
    y_min, y_max = 160, 320
    obs_in_region = [(x, y) for x, y in obs if x_min <= x <= x_max and y_min <= y <= y_max]
    # generate some random points in the upper region to place the points

    goal_states = [(random.randint(160, 480), random.randint(0, 140)) for _ in
                   range(len(obs_in_region))]

    # for all obstacles in the region, spawn a controller to remove it from the region
    processes = []
    i = 0

    print(goal_states)
    for obs in obs_in_region:
        start_state = np.asarray(obs)
        goal_state = np.asarray(goal_states[i])
        p = Process(target=ctrl,
                     args=(controls_parent, start_state, goal_state, spot_man))
        p.start()
        processes.append(p)
        i += 1

    for p in processes:
        p.join()

    spot_man.set_clearing_region(False)

def ctrl(controls_parent, start_state, goal_state, spot_man):
    spot_man.add_spot((int(start_state[0]), int(start_state[1])))
    obstacles = jnp.asarray(spot_man.get_obstacles())
    obstacles = obstacles[~jnp.all(obstacles == start_state, axis=1)]
    nearest_kps = []
    for obs in obstacles:
        dist = np.linalg.norm(np.array([start_state[0], start_state[1]]) - np.array(obs))
        if dist > 20:
            if dist < 250:
                nearest_kps.append(obs)

    obstacles = jnp.asarray(nearest_kps)

    current_length = obstacles.shape[0]
    if current_length < KPS_SIZE:
        padding_length = KPS_SIZE - current_length
        pad_array = jnp.zeros((padding_length, obstacles.shape[1]))
        obstacles = jnp.vstack([obstacles, pad_array])

    env = Environment.create(len(obstacles), obstacles)

    init_control = gen_initial_traj(start_state, goal_state, 35).T

    state = start_state
    dynamics = RK4Integrator(ContinuousTimeBeadDynamics(), DT)

    setattr(RunningCost, 'obstacle_separation', OBSTACLE_SEPARATION)
    time_step = 0
    while True:
        st = time.time()
        empty_env = Environment.create(0, jnp.array([]))

        solution = policy(state, goal_state, init_control, env, dynamics, RunningCost, MPCTerminalCost, empty_env,
                          False, 35)
        states, opt_controls = solution["optimal_trajectory"]
        control = opt_controls[0]

        #try:
        spot_man.move_trap((int(state[0]), int(state[1])), (int(control[0]), int(control[1])))
        #except:
            #print(f"Trap move out of bounds invalid: {state[0]}, {state[1]} to {control[0]}, {control[1]}")
        init_control = opt_controls
        state = control  # The control is the position of the bead (wherever we place the trap is wherever the bead will go)

        controls_parent.send(opt_controls)
        traps = []
        for pos in spot_man.get_trapped_beads().keys():
            traps.append(pos)

        kpsarray = jnp.asarray(traps)

        kpsarray = kpsarray[~jnp.all(kpsarray == np.array([int(control[0]), int(control[1])]), axis=1)]
        nearest_kps = []
        for kp in kpsarray:
            dist = np.linalg.norm(np.array([state[0], state[1]]) - np.array(kp))
            if dist > 20:
                if dist < 250:
                    nearest_kps.append(kp)

        kpsarray = jnp.asarray(nearest_kps)

        current_length = kpsarray.shape[0]
        if current_length == 0:
            print("AHA")
        if current_length < KPS_SIZE:
            padding_length = KPS_SIZE - current_length
            pad_array = jnp.zeros((padding_length, kpsarray.shape[1]))
            # pad_array = jnp.repeat(jnp.array([[state[0], state[1]]]), padding_length, axis=0)
            # print(pad_array)
            kpsarray = jnp.vstack([kpsarray, pad_array])

        env = env.update(kpsarray, len(kpsarray), time_step)

        dist_to_goal = np.sqrt((state[0] - goal_state[0]) ** 2 + (state[1] - goal_state[1]) ** 2)
        if dist_to_goal < 2:
            print("Goal reached!")
            break
        time_step += 1
        if keyboard.is_pressed('q'):
            break
        et = time.time()

        # rudimentary timing controller
        if 1 / (et - st) > 20 and 0.05 - (et - st) > 0:
            time.sleep(0.05 - (et - st))

if __name__ == "__main__":
    SpotManagerManager.register('SpotManager', SpotManager)
    SpotManagerManager.register('get_trapped_beads', SpotManager.get_trapped_beads)
    SpotManagerManager.register('add_spot', SpotManager.add_spot)
    SpotManagerManager.register('move_trap', SpotManager.move_trap)
    SpotManagerManager.register('remove_trap', SpotManager.remove_trap)
    SpotManagerManager.register('set_clearing_region', SpotManager.set_clearing_region)
    SpotManagerManager.register('get_clearing_region', SpotManager.get_clearing_region)
    SpotManagerManager.register('predict_mask', predict_mask)
    SpotManagerManager.register('init_phase_predictor', init_phase_predictor)

    manager = SpotManagerManager()
    manager.start()
    frame_queue = multiprocessing.Queue()  # Queue for camera frames
    controls_parent, controls_child = multiprocessing.Pipe() # Communicates future optimal controls to simulator

    spot_man = manager.SpotManager()
    lock = Lock()
    spot_lock = Lock()

    p1 = Process(target=simulator, args=(spot_man, controls_child, controls_parent))
    p2 = Process(target=init_holo_engine(), args=())
    p3 = Process(target=cam, args=(spot_man, controls_child, controls_parent))

    if SIMULATOR_MODE:
        p1.start()
    else:
        p3.start()

    #p2.start()

    if SIMULATOR_MODE:
        p1.join()
    else:
        p3.join()

    p2.join()


