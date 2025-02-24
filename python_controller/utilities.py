import os
import socket
import subprocess
import time
import numpy as np
import cv2
import math
import jax
import jax.numpy as jnp
import torch
from harvesters.core import Harvester
from constants import *
from multiprocessing.managers import BaseManager
from phase_prediction import mlp

class SpotManagerManager(BaseManager):
    pass

def create_artificial_obs(clustering, kps_array, frame):
    labels = clustering.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    labelled_clusters = np.concatenate((kps_array,labels[:,np.newaxis]),axis=1)
    artifical_pts = []
    # for each cluster found
    for i in range(num_clusters):
        filter = np.asarray([i])
        cluster_pts = labelled_clusters[np.in1d(labelled_clusters[:, -1], filter)]
        cluster_pts = cluster_pts[:, 0:-1]
        # for each point in cluster
        # draw a line from point 0 to point 1 then point 1 to point 2...
        start_pt = cluster_pts[0, :].astype(int)
        for j in range(len(cluster_pts) - 1):
            if DEBUG:
                cv2.line(frame, start_pt, cluster_pts[j+1, :].astype(int), (256, 0, 256), 1)
            line_dist = jnp.linalg.norm(start_pt - cluster_pts[j+1, :].astype(int))
            if line_dist > BEAD_RADIUS:
                n_pts = int((line_dist - BEAD_RADIUS) / (BEAD_RADIUS))
                interp_pts = np.linspace(start_pt, cluster_pts[j+1, :].astype(int), num=n_pts, endpoint=False)
                interp_pts = interp_pts[1:, :]
                artifical_pts.extend(interp_pts.tolist())
                if DEBUG:
                    for pt in interp_pts:
                        cv2.circle(frame, pt.astype(int), 13, (128, 0, 0), 1)
                start_pt = cluster_pts[j + 1, :].astype(int)

    return frame, artifical_pts

@jax.jit
def min_dist_to_ellipse(axis, pt):
    a, b = axis
    y0, y1 = pt

    # bisection bounds
    t0 = -(b ** 2) + b * y1
    t1 = -(b ** 2) + jnp.sqrt(((a ** 2) * (y0 ** 2)) + ((b ** 2) * (y1 ** 2)))

    # solve F to get roots
    def F(t, a, b, y0, y1):
        term0 = ((a * y0) / (t + a ** 2)) ** 2
        term1 = ((b * y1) / (t + b ** 2)) ** 2
        return term0 + term1 - 1

    def bisection_step(val, _):
        t0, t1 = val
        t_mid = (t0 + t1) / 2.0
        f_mid = F(t_mid, a, b, y0, y1)

        t0_new = jnp.where(f_mid > 0, t_mid, t0)
        t1_new = jnp.where(f_mid <= 0, t_mid, t1)
        return (t0_new, t1_new), None

    (t0_final, t1_final), _ = jax.lax.scan(bisection_step, (t0, t1), None, length=50)

    tbar = (t0_final + t1_final) / 2.0

    x0 = a * a * y0 / (tbar + a * a)
    x1 = b * b * y1 / (tbar + b * b)

    dist = jnp.sqrt((x0 - y0) ** 2 + (x1 - y1) ** 2)

    return dist

def mouse_callback(event, x, y, flags, param):
    spot_man, traps, dragging_trap_idx = param
    trapped_beads = []
    for pos in spot_man.get_trapped_beads().keys():
        trapped_beads.append(pos)

    line_trap = cv2.getTrackbarPos('LineTrap', 'Optical Tweezers Simulator')
    donut_trap = cv2.getTrackbarPos('DonutTrap', 'Optical Tweezers Simulator')

    # if line_trap and donut_trap:
    #     cv2.setTrackbarPos('LineTrap', 'Optical Tweezers Simulator', 0)

    mouse_pos = (x,y)
    if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_ALTKEY: # donut
        #donut_goal.append((x, y))
        spot_man.add_goal_pos(mouse_pos, is_donut=True)
        print("Donut Goal Added")
    elif event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY: # line
        #line_goal.append((x, y))
        spot_man.add_goal_pos(mouse_pos, is_line=True)
        print("Line Goal Added")
    elif event == cv2.EVENT_LBUTTONDOWN:  # Left click to add or select trap
        for i, trap in enumerate(traps):
            if np.linalg.norm(np.array([x, y]) - np.array(trap)) < 15:
                dragging_trap_idx[0] = i  # Start dragging this trap
                return
        # Add new trap if none selected
        if line_trap:
            spot_man.add_spot(mouse_pos, is_line=True)
            spot_man.add_start(mouse_pos, is_line=True)
        elif donut_trap:
            spot_man.add_spot((x,y), is_donut=True)
            #donut_start.append((x, y))
            spot_man.add_start(mouse_pos, is_donut=True)
        else:
            spot_man.add_spot((x,y))
            spot_man.add_start(mouse_pos)

        traps.append((x,y))
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right click to remove trap
        for j, trap in enumerate(traps):
            if np.linalg.norm(np.array([x,y]) - np.array(trap)) < 15:
                spot_man.remove_trap((trap[0], trap[1]))
                spot_man.remove_start((trap[0], trap[1]))
                traps.pop(j)
                return

        for k, trap in enumerate(trapped_beads):
            if np.linalg.norm(np.array([x,y]) - np.array(trap)) < 15:
                spot_man.remove_trap((trap[0], trap[1]))
                #spot_man.remove_start((trap[0], trap[1]))
                #traps.pop(k)
                return
        for goal in spot_man.get_goal_pos():
            if np.linalg.norm(np.array([x,y]) - np.array(goal)) < 15:
                spot_man.remove_goal_pos((goal[0], goal[1]))
                return
        for goal in spot_man.get_goal_pos(is_donut=True):
            if np.linalg.norm(np.array([x,y]) - np.array(goal)) < 15:
                spot_man.remove_goal_pos((goal[0], goal[1]))
                return
        for goal in spot_man.get_goal_pos(is_line=True):
            if np.linalg.norm(np.array([x,y]) - np.array(goal)) < 15:
                spot_man.remove_goal_pos((goal[0], goal[1]))
                return
    elif event == cv2.EVENT_MOUSEMOVE:  # Dragging trap around
        if dragging_trap_idx[0] is not None:
            old_pos = traps[dragging_trap_idx[0]]

            spot_man.move_trap(old_pos, mouse_pos)

            traps[dragging_trap_idx[0]] = mouse_pos

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
    elif event == cv2.EVENT_LBUTTONUP:  # Release dragging
        dragging_trap_idx[0] = None
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        # Todo: deselect a bead
        pass
    elif event == cv2.EVENT_MBUTTONDOWN:
        # add goal point
        spot_man.add_goal_pos(mouse_pos)

def init_holo_engine():
    """Initializes hologram engine by sending shader source code and updating uniform variables
    Communicates with the hologram engine via UDP packets
    """

    # Kill any instances of hologram engine
    os.system("taskkill /f /im  hologram_engine_64.exe")

    # Launch the new instance
    executable_path = os.path.join(os.getcwd(), "hologram_engine_64.exe")
    if os.path.exists(executable_path):
        holo_process = subprocess.Popen([executable_path])

    # subprocess.Popen([r'.\hologram_engine_64.exe'])

    # Define the path to the shader source file
    #shader_file_path = os.path.join('python_controller', 'shader_source.txt')
    # Define the path to the uniform variables file
    #uniform_vars_file_path = os.path.join('python_controller', 'init_uniform_vars.txt')
    shader_file_path = 'shader_source.txt'
    uniform_vars_file_path = 'init_uniform_vars.txt'
    time.sleep(1)
    with open(shader_file_path, 'r') as file:
        shader_source = file.read()
    with open(uniform_vars_file_path, 'r') as file:
        uniform_vars = file.read()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(('127.0.0.1', 61556))

    server_socket.sendto(str.encode(shader_source), ('127.0.0.1', 61557))
    server_socket.sendto(str.encode(uniform_vars), ('127.0.0.1', 61557))
    server_socket.close()
    return holo_process

def init_phase_predictor():
    """ Takes place of hologram engine, start opencv window and initialize pre-trained NN for phase mask predictions """
    model_path = r"C:\Users\tommyz\Desktop\Code\phase_prediction\models\mlp_model_5_spot_512-1024-2048-800k.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_model(model_path):
        model = mlp.MLP().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    model = load_model(model_path)

    return model


def predict_mask(spot_array, model):
    """ Input spot array, convert to n, 4, 4 array for model input """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = spot_array.shape[0]
    b = n // 16

    input_array = spot_array.reshape(b, 4, 4)
    if b < 5:
        pad_size = (5 - b, 4, 4)  # Compute required padding
        input_array = np.pad(input_array, ((0, pad_size[0]), (0, 0), (0, 0)), mode='constant', constant_values=0)

    spots_tensor = torch.tensor(input_array, dtype=torch.float32)
    spots_tensor = spots_tensor.unsqueeze(0)
    spots_tensor = spots_tensor.to(device)

    with torch.no_grad():  # Ensure no gradients are computed
        predicted_mask = model(spots_tensor).cpu().numpy()
    predicted_mask = predicted_mask[0]
    predicted_mask = (predicted_mask - predicted_mask.min()) / (predicted_mask.max() - predicted_mask.min()) * 255
    predicted_mask = predicted_mask.astype(np.uint8)

    cv2.namedWindow("Mask", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Mask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("Mask", 0, 0)
    cv2.resizeWindow("Mask", 512, 512)
    #print(np.max(predicted_mask))
    cv2.imshow("Mask", predicted_mask)
    cv2.waitKey(1)

def draw_traps(spot_man, frame):
    """
    Draws line, donut, and point traps on the displayed image
    """
    traps = spot_man.get_trapped_beads()
    goals = spot_man.get_goal_pos() #+ spot_man.get_goal_pos(is_line=True) + spot_man.get_goal_pos(is_donut=True)
    #virtual_traps = spot_man.get_virtual_traps()

    for donut_goal in spot_man.get_goal_pos(is_donut=True):
        cv2.circle(frame, (donut_goal[0], donut_goal[1]), 4, (0, 128, 256), -1)
        cv2.circle(frame, (donut_goal[0], donut_goal[1]), 8, (0, 128, 256), 1)
    for line_goal in spot_man.get_goal_pos(is_line=True):
        cv2.circle(frame, (line_goal[0], line_goal[1]), 4, (64, 128, 0), 1)

    for goal in goals:
        x, y = goal
        cv2.circle(frame, (x, y), 12, (64, 128, 0), 1)

    for key in traps:  # Draw traps
        x, y = key
        spot = traps.get(key)
        #sim_man.trap_bead(key)
        if spot.is_line:
            length = 60
            half_length = length / 2
            angle = spot.angle + (math.pi / 2)

            x_start = int(x - half_length * np.cos(angle))
            y_start = int(y - half_length * np.sin(angle))
            x_end = int(x + half_length * np.cos(angle))
            y_end = int(y + half_length * np.sin(angle))

            cv2.line(frame, (x_start, y_start), (x_end, y_end), (256, 0, 256), 1)
        elif spot.is_donut:
            cv2.circle(frame, (x, y), 23, (256, 0, 0), 1)
        else:
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