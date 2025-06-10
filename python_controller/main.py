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
    # Get initial obstacle positions
    kps = spot_man.get_obstacles()

    if is_donut:
        # Donut-shaped trap initialization
        start_pos = start
        spot_man.add_spot((start_pos[0], start_pos[1]), is_donut=True)
        x_start = float(start_pos[0])
        y_start = float(start_pos[1])
        goal_position = jnp.array([goal[0], goal[1]])
        N_n = N
        setattr(RunningCost, 'obstacle_separation', 1.2 * OBSTACLE_SEPARATION)
    elif is_line:
        # Line trap initialization
        start_pos = start
        spot_man.add_spot((start_pos[0], start_pos[1]), is_line=True)
        x_start = float(start_pos[0])
        y_start = float(start_pos[1])
        goal_position = jnp.array([goal[0], goal[1]])
        N_n = N
        setattr(RunningCost, 'obstacle_separation', 1 * OBSTACLE_SEPARATION)
        setattr(RunningCost, 'agent_as_ellipse', True)
        setattr(RunningCost, 'ellipse_axis', [ELLIPSE_MAJOR_AXIS, ELLIPSE_MINOR_AXIS])
    else:
        # Default single point trap
        x_start = float(start[0])
        y_start = float(start[1])
        goal_position = jnp.array([goal[0], goal[1]])
        N_n = N
        setattr(RunningCost, 'obstacle_separation', OBSTACLE_SEPARATION)

    start_state = jnp.array([x_start, y_start])

    # Remove start location from obstacle list
    kpsarray = jnp.asarray(kps)
    kpsarray = kpsarray[~jnp.all(kpsarray == start_state, axis=1)]

    # Pad obstacle list if too short (JAX shape constraints)
    current_length = kpsarray.shape[0]
    if current_length < KPS_SIZE:
        padding_length = KPS_SIZE - current_length
        pad_array = jnp.zeros((padding_length, kpsarray.shape[1]))
        kpsarray = jnp.vstack([kpsarray, pad_array])

    # Initialize environment and trajectory
    env = Environment.create(len(kpsarray), kpsarray)
    init_control = gen_initial_traj(start_state, goal_position, N_n).T
    state = start_state
    dynamics = RK4Integrator(ContinuousTimeBeadDynamics(), DT)
    time_step = 0
    ctrl_idx = 0
    empty_env = Environment.create(0, jnp.array([]))

    while True:
        st = time.time()

        # Recompute MPC trajectory at fixed intervals
        if time_step % MPC_COMPUTE_FREQ == 0:
            solution = policy(state, goal_position, init_control, env, dynamics, RunningCost, MPCTerminalCost, empty_env, False, N_n)
            states, opt_controls = solution["optimal_trajectory"]

        # Control index reset or increment
        if time_step % MPC_COMPUTE_FREQ == 0:
            ctrl_idx = 0
        else:
            ctrl_idx += 1

        control = opt_controls[ctrl_idx]

        try:
            # Physically move trap
            spot_man.move_trap((int(state[0]), int(state[1])), (int(control[0]), int(control[1])))
        except:
            print(f"Trap move out of bounds invalid: {state[0]}, {state[1]} to {control[0]}, {control[1]}")
            return

        # Update control initialization for next round
        init_control = opt_controls
        state = control  # Bead follows trap

        # Send current control sequence to parent
        controls_parent.send(opt_controls)

        # Update obstacle list
        kpsarray = jnp.asarray(spot_man.get_obstacles())
        kpsarray = kpsarray[~jnp.all(kpsarray == np.array([int(control[0]), int(control[1])]), axis=1)]

        # Optionally remove trapped beads
        kpsarray = remove_trapped_beads(state, kpsarray, is_donut, is_line)

        # Remove nearby obstacles if close to goal
        dist_to_goal = np.sqrt((state[0] - goal_position[0]) ** 2 + (state[1] - goal_position[1]) ** 2)
        if dist_to_goal < GOAL_DIST_OBSTACLE_FREE and kpsarray.size != 0:
            distances = np.linalg.norm(kpsarray - goal_position, axis=1)
            kpsarray = kpsarray[distances > 75]

        # Pad obstacle array again if needed
        current_length = kpsarray.shape[0]
        if current_length < KPS_SIZE:
            padding_length = KPS_SIZE - current_length
            if kpsarray.size == 0:
                kpsarray = jnp.zeros((padding_length, 2))
            else:
                pad_array = jnp.zeros((padding_length, kpsarray.shape[1]))
                kpsarray = jnp.vstack([kpsarray, pad_array])

        # Update environment with new obstacle set
        env = env.update(kpsarray, len(kpsarray), time_step)

        # Goal reached condition
        if dist_to_goal < 2:
            print("Goal reached!")
            break

        # User-initiated break
        if keyboard.is_pressed('q'):
            break

        et = time.time()
        time_step += 1

        # Enforce max update rate (~20 Hz)
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
    dt = 0.0015  # Simulation time step
    dragging_trap_idx = [None]

    # Randomly initialize bead positions
    for _ in range(number_of_beads):
        x_start = random.randint(0, CAM_X - 1)
        y_start = random.randint(0, CAM_Y - 1)
        sim_man.add_bead((x_start, y_start))

    traps, donut_start, donut_goal, line_start, line_goal = [], [], [], [], []
    params = [spot_man, traps, dragging_trap_idx]

    def nothing(x):
        pass

    # Set up GUI window and controls
    cv2.namedWindow("Optical Tweezers Simulator")
    cv2.setMouseCallback("Optical Tweezers Simulator", mouse_callback, param=params)
    cv2.createTrackbar('LineTrap', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('DonutTrap', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('PointGoal', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('MoveDonutLineToGoal', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('ClearObs', 'Optical Tweezers Simulator', 0, 1, nothing)

    dynamics = ContinuousTimeObstacleDynamics()

    while True:
        # Read GUI control states
        ctrl_zero = cv2.getTrackbarPos('PointGoal', 'Optical Tweezers Simulator')
        ctrl_one = cv2.getTrackbarPos('MoveDonutLineToGoal', 'Optical Tweezers Simulator')
        ctrl_two = cv2.getTrackbarPos('ClearObs', 'Optical Tweezers Simulator')
        line_trap = cv2.getTrackbarPos('LineTrap', 'Optical Tweezers Simulator')
        donut_trap = cv2.getTrackbarPos('DonutTrap', 'Optical Tweezers Simulator')

        # Prevent both line and donut trap from being active
        if line_trap and donut_trap:
            cv2.setTrackbarPos('LineTrap', 'Optical Tweezers Simulator', 0)

        # Trigger point-goal MPC trajectories
        if ctrl_zero:
            point_starts = spot_man.get_start_pos()
            point_goals = spot_man.get_goal_pos()
            for start_pos in point_starts:
                if len(point_goals) != 0:
                    goal_pos = point_goals.popleft()
                    p = Process(target=holo, args=(controls_parent, spot_man, goal_pos, start_pos))
                    p.start()
            spot_man.clear_starts()
            cv2.setTrackbarPos('PointGoal', 'Optical Tweezers Simulator', 0)

        # Trigger donut/line MPC trajectories
        elif ctrl_one:
            donut_starts = spot_man.get_start_pos(is_donut=True)
            line_starts = spot_man.get_start_pos(is_line=True)
            donut_goals = spot_man.get_goal_pos(is_donut=True)
            line_goals = spot_man.get_goal_pos(is_line=True)

            for start_pos in donut_starts:
                goal_pos = donut_goals.popleft()
                p = Process(target=holo, args=(controls_parent, spot_man, goal_pos, start_pos, True, False))
                p.start()

            for start_pos in line_starts:
                goal_pos = line_goals.popleft()
                p = Process(target=holo, args=(controls_parent, spot_man, goal_pos, start_pos, False, True))
                p.start()

            spot_man.clear_starts(is_donut=True)
            spot_man.clear_starts(is_line=True)
            cv2.setTrackbarPos('MoveDonutLineToGoal', 'Optical Tweezers Simulator', 0)

        # Trigger obstacle clearing process
        elif ctrl_two:
            if not spot_man.get_clearing_region():
                p = Process(target=clear_region, args=(controls_parent, spot_man))
                p.start()
            cv2.setTrackbarPos('ClearObs', 'Optical Tweezers Simulator', 0)

        frame = white_bg.copy()

        # Move beads using Brownian motion
        sim_man.brownian_move(dt, dynamics)

        # Draw current traps
        frame = draw_traps(spot_man, frame)

        # Receive and draw predicted control points (optional)
        if controls_child.poll():
            opt_controls = controls_child.recv().reshape(-1, 2)
            for g, cont in enumerate(opt_controls):
                if g % 5 == 0 and DEBUG:
                    cv2.circle(frame, (int(cont[0]), int(cont[1])), 2, (128, 0, 0), -1)

        # Detect beads from simulated image
        key_points = camera.detect_beads(frame, is_simulator=True)
        if DEBUG:
            cv2.drawKeypoints(frame, key_points, frame, (255, 0, 0))

        points = [[kp.pt[0], kp.pt[1]] for kp in key_points]
        kps_array = np.asarray(points)

        # Cluster and generate artificial obstacles
        clustering = DBSCAN(eps=60, min_samples=2, n_jobs=-1).fit(kps_array)
        frame, artifical_pts = create_artificial_obs(clustering, kps_array, frame)

        # Update obstacle information
        spot_man.set_obstacles(points)
        spot_man.set_fake_obstacles(artifical_pts)

        # Display frame
        cv2.imshow("Optical Tweezers Simulator", frame)

        # Exit on 'q' key press
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

    # Setup GUI window and controls
    cv2.namedWindow("Optical Tweezers Simulator")
    cv2.setMouseCallback("Optical Tweezers Simulator", mouse_callback, param=params)
    cv2.createTrackbar('LineTrap', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('DonutTrap', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('PointGoal', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('MoveDonutLineToGoal', 'Optical Tweezers Simulator', 0, 1, nothing)
    cv2.createTrackbar('ClearObs', 'Optical Tweezers Simulator', 0, 1, nothing)

    while True:
        # Acquire camera frame
        with ia.fetch() as buffer:
            component = buffer.payload.components[0]
            img = np.ndarray(buffer=component.data.copy(), dtype=np.uint8,
                             shape=(component.height, component.width, 1))

        # Detect beads
        key_points = camera.detect_beads(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read GUI controls
        ctrl_zero = cv2.getTrackbarPos('PointGoal', 'Optical Tweezers Simulator')
        ctrl_one = cv2.getTrackbarPos('MoveDonutLineToGoal', 'Optical Tweezers Simulator')
        ctrl_two = cv2.getTrackbarPos('ClearObs', 'Optical Tweezers Simulator')
        line_trap = cv2.getTrackbarPos('LineTrap', 'Optical Tweezers Simulator')
        donut_trap = cv2.getTrackbarPos('DonutTrap', 'Optical Tweezers Simulator')

        # Prevent both trap types from being active simultaneously
        if line_trap and donut_trap:
            cv2.setTrackbarPos('LineTrap', 'Optical Tweezers Simulator', 0)

        # Launch MPC for point-goal traps
        if ctrl_zero:
            point_starts = spot_man.get_start_pos()
            point_goals = spot_man.get_goal_pos()
            for start_pos in point_starts:
                if len(point_goals) != 0:
                    goal_pos = point_goals.popleft()
                    p = Process(target=holo, args=(controls_parent, spot_man, goal_pos, start_pos))
                    p.start()
            spot_man.clear_starts()
            cv2.setTrackbarPos('PointGoal', 'Optical Tweezers Simulator', 0)

        # Launch MPC for donut and line traps
        elif ctrl_one:
            donut_starts = spot_man.get_start_pos(is_donut=True)
            line_starts = spot_man.get_start_pos(is_line=True)
            donut_goals = spot_man.get_goal_pos(is_donut=True)
            line_goals = spot_man.get_goal_pos(is_line=True)

            for start_pos in donut_starts:
                goal_pos = donut_goals.popleft()
                p = Process(target=holo, args=(controls_parent, spot_man, goal_pos, start_pos, True, False))
                p.start()

            for start_pos in line_starts:
                goal_pos = line_goals.popleft()
                p = Process(target=holo, args=(controls_parent, spot_man, goal_pos, start_pos, False, True))
                p.start()

            spot_man.clear_starts(is_donut=True)
            spot_man.clear_starts(is_line=True)
            cv2.setTrackbarPos('MoveDonutLineToGoal', 'Optical Tweezers Simulator', 0)

        # Trigger region-clearing routine
        elif ctrl_two:
            if not spot_man.get_clearing_region():
                p = Process(target=clear_region, args=(controls_parent, spot_man))
                p.start()
            cv2.setTrackbarPos('ClearObs', 'Optical Tweezers Simulator', 0)

        # Draw active traps
        img = draw_traps(spot_man, img)

        # Optional: show predicted control output from policy
        if controls_child.poll():
            opt_controls = controls_child.recv().reshape(-1, 2)
            # Optional debug drawing of control points

        # Optional: draw keypoints for visualization
        if DEBUG:
            cv2.drawKeypoints(img, key_points, img, (255, 0, 0))
        cv2.imshow('Optical Tweezers Simulator', img)

        # Update obstacle list for policy input
        points = [[kp.pt[0], kp.pt[1]] for kp in key_points]
        kps_array = np.asarray(points)

        # Optional DBSCAN artificial obstacle generation
        # if len(kps_array) > 2:
        #     clustering = DBSCAN(eps=60, min_samples=2).fit(kps_array)
        #     img, artifical_pts = create_artificial_obs(clustering, kps_array, img)
        #     combined_pts = points + artifical_pts
        # else:
        combined_pts = points

        spot_man.set_obstacles(combined_pts)

        cv2.waitKey(1)
        k += 1

        # Exit on 'q' key press
        if keyboard.is_pressed('q'):
            ia.stop()
            ia.destroy()
            os.system("taskkill /f /im  hologram_engine_64.exe")
            break

def clear_region(controls_parent, spot_man):
    """
    Move all detected obstacles in a specified region (160–480 x, 160–320 y)
    to a randomly chosen area outside that region to clear the workspace.
    """
    obs = spot_man.get_obstacles()
    spot_man.set_clearing_region(True)

    # Define rectangular region to clear
    x_min, x_max = 160, 480
    y_min, y_max = 160, 320
    obs_in_region = [(x, y) for x, y in obs if x_min <= x <= x_max and y_min <= y <= y_max]

    # Generate random target positions above the region
    goal_states = [(random.randint(160, 480), random.randint(0, 140)) for _ in obs_in_region]

    processes = []

    for i, obs in enumerate(obs_in_region):
        start_state = np.asarray(obs)
        goal_state = np.asarray(goal_states[i])
        p = Process(target=ctrl, args=(controls_parent, start_state, goal_state, spot_man))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    spot_man.set_clearing_region(False)

def ctrl(controls_parent, start_state, goal_state, spot_man):
    """
    Use iLQR-based MPC to guide a trapped bead (obstacle) from start_state to goal_state
    while avoiding nearby keypoints (obstacles).
    """
    spot_man.add_spot((int(start_state[0]), int(start_state[1])))

    # Filter out nearby obstacles
    obstacles = jnp.asarray(spot_man.get_obstacles())
    obstacles = obstacles[~jnp.all(obstacles == start_state, axis=1)]

    # Keep only relevant nearby obstacles for avoidance
    nearest_kps = [obs for obs in obstacles if 20 < np.linalg.norm(start_state - obs) < 250]
    obstacles = jnp.asarray(nearest_kps)

    # Pad obstacle list if needed
    current_length = obstacles.shape[0]
    if current_length < KPS_SIZE:
        padding = jnp.zeros((KPS_SIZE - current_length, 2))
        obstacles = jnp.vstack([obstacles, padding])

    env = Environment.create(len(obstacles), obstacles)
    init_control = gen_initial_traj(start_state, goal_state, 35).T

    state = start_state
    dynamics = RK4Integrator(ContinuousTimeBeadDynamics(), DT)

    setattr(RunningCost, 'obstacle_separation', OBSTACLE_SEPARATION)
    time_step = 0

    while True:
        st = time.time()
        empty_env = Environment.create(0, jnp.array([]))

        solution = policy(
            state, goal_state, init_control, env, dynamics,
            RunningCost, MPCTerminalCost, empty_env,
            static=True, horizon=35
        )

        states, opt_controls = solution["optimal_trajectory"]
        control = opt_controls[0]

        # Apply control to trap
        spot_man.move_trap((int(state[0]), int(state[1])), (int(control[0]), int(control[1])))
        init_control = opt_controls
        state = control

        # Share control with parent process
        controls_parent.send(opt_controls)

        # Update environment with other currently trapped beads
        traps = list(spot_man.get_trapped_beads().keys())
        kpsarray = jnp.asarray(traps)
        kpsarray = kpsarray[~jnp.all(kpsarray == state, axis=1)]

        nearest_kps = [kp for kp in kpsarray if 20 < np.linalg.norm(state - kp) < 250]
        kpsarray = jnp.asarray(nearest_kps)

        if kpsarray.shape[0] < KPS_SIZE:
            padding = jnp.zeros((KPS_SIZE - kpsarray.shape[0], 2))
            kpsarray = jnp.vstack([kpsarray, padding])

        env = env.update(kpsarray, len(kpsarray), time_step)

        # Stop when close enough to goal
        if np.linalg.norm(state - goal_state) < 2:
            print("Goal reached!")
            break

        time_step += 1

        # Manual interrupt
        if keyboard.is_pressed('q'):
            break

        # Maintain control loop frequency ~20Hz
        et = time.time()
        delay = max(0, 0.05 - (et - st))
        if delay > 0:
            time.sleep(delay)

if __name__ == "__main__":
    # Register methods for remote access via the SpotManagerManager proxy
    SpotManagerManager.register('SpotManager', SpotManager)
    SpotManagerManager.register('get_trapped_beads', SpotManager.get_trapped_beads)
    SpotManagerManager.register('add_spot', SpotManager.add_spot)
    SpotManagerManager.register('move_trap', SpotManager.move_trap)
    SpotManagerManager.register('remove_trap', SpotManager.remove_trap)
    SpotManagerManager.register('set_clearing_region', SpotManager.set_clearing_region)
    SpotManagerManager.register('get_clearing_region', SpotManager.get_clearing_region)

    # Launch manager
    manager = SpotManagerManager()
    manager.start()

    # Shared communication channels
    controls_parent, controls_child = multiprocessing.Pipe()  # For sending MPC outputs

    # Shared state manager for traps
    spot_man = manager.SpotManager()

    # Locks for thread-safe access if needed later
    lock = Lock()
    spot_lock = Lock()

    # Define subprocesses
    p1 = Process(target=simulator, args=(spot_man, controls_child, controls_parent))  # MPC simulation mode
    p2 = Process(target=init_holo_engine)  # Launches hologram rendering backend
    p3 = Process(target=cam, args=(spot_man, controls_child, controls_parent))  # Main camera loop

    # Choose between real-time camera or simulation
    if SIMULATOR_MODE:
        p1.start()
    else:
        p3.start()

    p2.start()  # Always start hologram engine

    # Wait for selected main loop to exit
    if SIMULATOR_MODE:
        p1.join()
    else:
        p3.join()

    p2.join()
