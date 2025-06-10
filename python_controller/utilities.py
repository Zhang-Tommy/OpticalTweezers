import os
import socket
import subprocess
import time
import numpy as np
import cv2
import math
import jax
import jax.numpy as jnp
from harvesters.core import Harvester
from constants import *
from multiprocessing.managers import BaseManager

class SpotManagerManager(BaseManager):
    pass

def create_artificial_obs(clustering, kps_array, frame):
    labels = clustering.labels_  # Cluster labels for each keypoint
    # Count number of clusters, ignoring noise label (-1)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Append cluster labels as an additional column to keypoints array
    labelled_clusters = np.concatenate((kps_array, labels[:, np.newaxis]), axis=1)
    artifical_pts = []

    # Iterate through each cluster (excluding noise)
    for i in range(num_clusters):
        filter = np.asarray([i])
        # Select points belonging to the current cluster
        cluster_pts = labelled_clusters[np.in1d(labelled_clusters[:, -1], filter)]
        cluster_pts = cluster_pts[:, 0:-1]  # Remove label column, keep only coordinates

        # Initialize starting point for line drawing as first point in cluster
        start_pt = cluster_pts[0, :].astype(int)

        # Iterate over consecutive pairs of points in the cluster
        for j in range(len(cluster_pts) - 1):
            # If DEBUG mode is on, draw a line between consecutive points
            if DEBUG:
                cv2.line(frame, start_pt, cluster_pts[j + 1, :].astype(int), (256, 0, 256), 1)

            # Compute Euclidean distance between start_pt and next point
            line_dist = jnp.linalg.norm(start_pt - cluster_pts[j + 1, :].astype(int))

            # If the distance is greater than BEAD_RADIUS, interpolate points between
            if line_dist > BEAD_RADIUS:
                # Number of points to interpolate along the line, excluding endpoint
                n_pts = int((line_dist - BEAD_RADIUS) / BEAD_RADIUS)

                # Create equally spaced points between start_pt and next cluster point
                interp_pts = np.linspace(start_pt, cluster_pts[j + 1, :].astype(int), num=n_pts, endpoint=False)
                interp_pts = interp_pts[1:, :]  # Skip the first point which is start_pt itself

                # Add interpolated points to artificial obstacle points list
                artifical_pts.extend(interp_pts.tolist())

                # If DEBUG mode is on, draw circles at interpolated points
                if DEBUG:
                    for pt in interp_pts:
                        cv2.circle(frame, pt.astype(int), 13, (128, 0, 0), 1)

            # Update start_pt to current point for next iteration
            start_pt = cluster_pts[j + 1, :].astype(int)

    return frame, artifical_pts

@jax.jit
def min_dist_to_ellipse(axis, pt):
    a, b = axis  # ellipse semi-axes lengths
    y0, y1 = pt  # point coordinates

    # Initial bisection interval bounds for parameter t
    t0 = -(b ** 2) + b * y1
    t1 = -(b ** 2) + jnp.sqrt((a ** 2) * (y0 ** 2) + (b ** 2) * (y1 ** 2))

    # Function F(t) whose root defines closest point on ellipse to pt
    def F(t, a, b, y0, y1):
        term0 = ((a * y0) / (t + a ** 2)) ** 2
        term1 = ((b * y1) / (t + b ** 2)) ** 2
        return term0 + term1 - 1  # zero when point lies on ellipse parameterized by t

    # One bisection iteration step to narrow root interval of F(t)
    def bisection_step(val, _):
        t0, t1 = val
        t_mid = (t0 + t1) / 2.0
        f_mid = F(t_mid, a, b, y0, y1)

        # If F(t_mid) > 0, root is in [t_mid, t1], else in [t0, t_mid]
        t0_new = jnp.where(f_mid > 0, t_mid, t0)
        t1_new = jnp.where(f_mid <= 0, t_mid, t1)
        return (t0_new, t1_new), None

    # Perform 50 iterations of bisection to approximate root tbar
    (t0_final, t1_final), _ = jax.lax.scan(bisection_step, (t0, t1), None, length=50)
    tbar = (t0_final + t1_final) / 2.0

    # Compute closest point on ellipse to pt using tbar
    x0 = a * a * y0 / (tbar + a * a)
    x1 = b * b * y1 / (tbar + b * b)

    # Euclidean distance between pt and closest ellipse point
    dist = jnp.sqrt((x0 - y0) ** 2 + (x1 - y1) ** 2)

    return dist

def mouse_callback(event, x, y, flags, param):
    spot_man, traps, dragging_trap_idx = param

    # Get list of currently trapped bead positions
    trapped_beads = []
    for pos in spot_man.get_trapped_beads().keys():
        trapped_beads.append(pos)

    # Read current trap mode from GUI trackbars
    line_trap = cv2.getTrackbarPos('LineTrap', 'Optical Tweezers Simulator')
    donut_trap = cv2.getTrackbarPos('DonutTrap', 'Optical Tweezers Simulator')

    mouse_pos = (x, y)

    # ALT + Left Click: Add a donut-shaped goal position
    if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_ALTKEY:
        spot_man.add_goal_pos(mouse_pos, is_donut=True)
        print("Donut Goal Added")

    # CTRL + Left Click: Add a line-shaped goal position
    elif event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
        spot_man.add_goal_pos(mouse_pos, is_line=True)
        print("Line Goal Added")

    # Left Click without modifier: Add a new trap or start dragging an existing trap
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is near any existing trap to start dragging
        for i, trap in enumerate(traps):
            if np.linalg.norm(np.array([x, y]) - np.array(trap)) < 15:
                dragging_trap_idx[0] = i  # Begin dragging this trap
                return
        # If no trap selected, add a new trap and corresponding start point
        if line_trap:
            spot_man.add_spot(mouse_pos, is_line=True)
            spot_man.add_start(mouse_pos, is_line=True)
        elif donut_trap:
            spot_man.add_spot(mouse_pos, is_donut=True)
            spot_man.add_start(mouse_pos, is_donut=True)
        else:
            spot_man.add_spot(mouse_pos)
            spot_man.add_start(mouse_pos)
        traps.append(mouse_pos)

    # Right Click: Remove a trap, trapped bead, or goal if click is near any
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Remove trap and start if click is near an existing trap
        for j, trap in enumerate(traps):
            if np.linalg.norm(np.array([x, y]) - np.array(trap)) < 15:
                spot_man.remove_trap(trap)
                spot_man.remove_start(trap)
                traps.pop(j)
                return
        # Remove trapped bead's trap if click is near a trapped bead
        for k, trap in enumerate(trapped_beads):
            if np.linalg.norm(np.array([x, y]) - np.array(trap)) < 15:
                spot_man.remove_trap(trap)
                return
        # Remove goal points near the click position
        for goal in spot_man.get_goal_pos():
            if np.linalg.norm(np.array([x, y]) - np.array(goal)) < 15:
                spot_man.remove_goal_pos(goal)
                return
        for goal in spot_man.get_goal_pos(is_donut=True):
            if np.linalg.norm(np.array([x, y]) - np.array(goal)) < 15:
                spot_man.remove_goal_pos(goal)
                return
        for goal in spot_man.get_goal_pos(is_line=True):
            if np.linalg.norm(np.array([x, y]) - np.array(goal)) < 15:
                spot_man.remove_goal_pos(goal)
                return

    # Mouse move: Drag the currently selected trap to new position
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_trap_idx[0] is not None:
            old_pos = traps[dragging_trap_idx[0]]

            spot_man.move_trap(old_pos, mouse_pos)  # Update trap position

            traps[dragging_trap_idx[0]] = mouse_pos  # Update local trap list

            # Update corresponding start position depending on trap type
            if old_pos in spot_man.get_start_pos(is_line=True):
                spot_man.remove_start(old_pos)
                spot_man.add_start(mouse_pos, is_line=True)
            elif old_pos in spot_man.get_start_pos(is_donut=True):
                spot_man.remove_start(old_pos)
                spot_man.add_start(mouse_pos, is_donut=True)
            else:
                spot_man.remove_start(old_pos)
                spot_man.add_start(mouse_pos)
            return

    # Left Button Release: Stop dragging the trap
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_trap_idx[0] = None

    # Double Right Click: (TODO) Deselect a bead - not implemented
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        pass

    # Middle Button Click: Add a goal point at mouse position
    elif event == cv2.EVENT_MBUTTONDOWN:
        spot_man.add_goal_pos(mouse_pos)

def init_holo_engine():
    """
    Initializes the hologram engine by launching the executable,
    sending shader source code and uniform variables via UDP packets.
    """

    # Kill any running instances of the hologram engine to avoid conflicts
    os.system("taskkill /f /im hologram_engine_64.exe")

    # Launch a new instance of the hologram engine executable if it exists
    executable_path = os.path.join(os.getcwd(), "hologram_engine_64.exe")
    if os.path.exists(executable_path):
        holo_process = subprocess.Popen([executable_path])

    # Wait briefly to ensure the hologram engine is up and listening
    time.sleep(1)

    # Read the shader source code from file
    shader_file_path = 'shader_source.txt'
    with open(shader_file_path, 'r') as file:
        shader_source = file.read()

    # Read the initial uniform variables from file
    uniform_vars_file_path = 'init_uniform_vars.txt'
    with open(uniform_vars_file_path, 'r') as file:
        uniform_vars = file.read()

    # Set up UDP socket for communication with the hologram engine
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(('127.0.0.1', 61556))  # Bind to local port

    # Send shader source code and uniform variables to the hologram engine
    server_socket.sendto(str.encode(shader_source), ('127.0.0.1', 61557))
    server_socket.sendto(str.encode(uniform_vars), ('127.0.0.1', 61557))

    # Close the socket after sending data
    server_socket.close()

    return holo_process  # Return the process handle for possible later use

def draw_traps(spot_man, frame):
    """
    Draws line, donut, and point traps on the given frame image.
    """

    # Get currently trapped beads and goal positions
    traps = spot_man.get_trapped_beads()
    goals = spot_man.get_goal_pos()

    # Draw donut trap goals as filled and outlined circles
    for donut_goal in spot_man.get_goal_pos(is_donut=True):
        cv2.circle(frame, (donut_goal[0], donut_goal[1]), 4, (0, 128, 256), -1)
        cv2.circle(frame, (donut_goal[0], donut_goal[1]), 8, (0, 128, 256), 1)

    # Draw line trap goals as small outlined circles
    for line_goal in spot_man.get_goal_pos(is_line=True):
        cv2.circle(frame, (line_goal[0], line_goal[1]), 4, (64, 128, 0), 1)

    # Draw all goals as larger outlined circles
    for goal in goals:
        x, y = goal
        cv2.circle(frame, (x, y), 12, (64, 128, 0), 1)

    # Draw the traps themselves
    for key in traps:
        x, y = key
        spot = traps.get(key)

        if spot.is_line:
            # Draw line trap as a short line centered on (x, y) with given angle
            length = 60
            half_length = length / 2
            angle = spot.angle + (math.pi / 2)

            x_start = int(x - half_length * np.cos(angle))
            y_start = int(y - half_length * np.sin(angle))
            x_end = int(x + half_length * np.cos(angle))
            y_end = int(y + half_length * np.sin(angle))

            cv2.line(frame, (x_start, y_start), (x_end, y_end), (256, 0, 256), 1)
        elif spot.is_donut:
            # Draw donut trap as a circle with radius 23
            cv2.circle(frame, (x, y), 23, (256, 0, 0), 1)
        else:
            # Draw point trap as a smaller circle
            cv2.circle(frame, (x, y), 12, (0, 256, 0), 1)

    return frame

def start_image_acquisition():
    """ Initializes the gigE camera"""
    h = Harvester()
    h.add_file(CTI_FILE_DIR)
    h.update()

    ia = h.create()
    ia.start()

    return ia

def remove_closest_point(coords, x, y):
    """ Removes the closest point to (x,y) in the coords array"""
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

def sync_pipes(one, two, parent):
    """ Buffer for pipes, takes multiple inputs and outputs a single concatenated list"""
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

def send_data(message):
    """ Send UDP packet containing message to hologram engine"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    start = '<data>\n'
    end = '\n</data>'
    server_socket.sendto(str.encode(start + message + end), (UDP_IP, UDP_PORT))
    server_socket.close()

def set_uniform(var_num, new_data):
    """ Set specific uniform variables in hologram engine (configuration related) """
    start = f'<uniform id={var_num}>\n'
    end = '\n</uniform>'
    send_data(start + new_data + end)

def cam_to_um(cam_pos):
    """ Converts from camera coordinates to micrometers in workspace """
    um_x = cam_pos[0] * CAM_TO_UM
    um_y = cam_pos[1] * CAM_TO_UM
    return [um_x, um_y]

def kill_holo_engine():
    """Terminate the hologram engine executable"""
    os.system("taskkill /f /im  hologram_engine_64.exe")