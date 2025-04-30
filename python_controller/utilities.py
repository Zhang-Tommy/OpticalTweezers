import os
import socket
import subprocess
import time
from cmath import phase

import numpy as np
import cv2
import jax
import jax.numpy as jnp
import torch
from harvesters.core import Harvester
from constants import *
from multiprocessing.managers import BaseManager
import read_lut
from red_tweezers import calculate_phase_mask
from collections import OrderedDict
from phase_prediction import unet
from phase_prediction import input_converter
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

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
    executable_path = os.path.join(os.getcwd(), "dependencies/hologram_engine_64.exe")
    if os.path.exists(executable_path):
        holo_process = subprocess.Popen([executable_path])

    # subprocess.Popen([r'.\hologram_engine_64.exe'])

    # Define the path to the shader source file
    #shader_file_path = os.path.join('python_controller', 'shader_source.txt')
    # Define the path to the uniform variables file
    #uniform_vars_file_path = os.path.join('python_controller', 'init_uniform_vars.txt')
    shader_file_path = 'dependencies/shader_source.txt'
    uniform_vars_file_path = 'dependencies/init_uniform_vars.txt'
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model("./phase_prediction/best_unet_256_200k_4_30_2025.pth", device)
    return model

def load_model(model_path, device):
    model = unet.UNet(in_channels=1, out_channels=1, init_features=64).to(device)
    state_dict = torch.load(model_path, map_location=device)

    # Remove "module." prefix if trained with DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def calculate_farfield(phase_mask):
    def gaussian_beam(mask_size, beam_width):
        x = np.linspace(-1, 1, mask_size)
        y = np.linspace(-1, 1, mask_size)
        xx, yy = np.meshgrid(x, y)
        return np.exp(-(xx ** 2 + yy ** 2) / (2 * beam_width ** 2))

    def compute_beam_width(mask_sz, reference_size=512, reference_bm=0.05):
        return reference_bm * (reference_size / mask_sz)

    mask_size = 512
    bm = compute_beam_width(mask_size)
    incident_beam = gaussian_beam(mask_size, beam_width=bm)

    slm_field = incident_beam * np.exp(1j * phase_mask)
    far_field = fftshift(fft2(slm_field)) / ((2 * mask_size) ** 2)
    intensity = np.abs(far_field) ** (1 / 2)
    intensity_farfield = intensity / np.max(intensity)

    return intensity_farfield

def predict_mask(spot_array, model):
    n = spot_array.shape[0] // 16 # number of spots
    spot_array = np.reshape(spot_array, (n, 4, 4))

    # Calculate predicted phase mask
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    intensity_mask = input_converter.gen_input_intensity(spot_array)
    intensity_input = torch.tensor(intensity_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    prediction = model(intensity_input.to(device))
    prediction = prediction.detach().cpu().numpy().squeeze(0).squeeze(0)
    prediction = np.rot90(prediction, 1)
    prediction = np.flip(prediction, 0)

    prediction_resized = jax.image.resize(prediction * np.pi,(512, 512), 'nearest')


    predicted_farfield = calculate_farfield(prediction_resized)

    def to_cv(np_array):
        return ((np_array - np_array.min()) / (np_array.max() - np_array.min()) * 255).astype(np.uint8)

    window_width = 512
    window_height = 512
    windows = [
        ("Phase Mask", to_cv(prediction), (0, 0)),
        ("Unwrapped Phase Mask", to_cv(predicted_farfield), (window_width, 0)),
        #("Far Field", intensity_farfield, (window_width * 2, 0)),
        #("Unwrapped Far Field", intensity_farfield_u, (window_width * 3, 0)),
        #("Target Mask", intensity_mask, (window_width * 4, 0)),
    ]

    for title, image, pos in windows:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, window_width, window_height)
        cv2.moveWindow(title, *pos)
        cv2.imshow(title, image)

    cv2.waitKey(1)  # Use waitKey(0) to pause until keypress




def predict_mask1(spot_array, model):
    """ Input spot array, convert to n, 4, 4 array for model input """
    n = spot_array.shape[0] # number of spots * 16
    spot_array = np.reshape(spot_array, (n // 16, 4, 4))

    mask_type = 'lg'
    #if mask_type == 'lg':
    phase_mask = calculate_phase_mask(spot_array, n, 256, False)[0]

    intensity_mask = input_converter.gen_input_intensity(spot_array)
    #intensity_input = torch.tensor(intensity_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # lenses and gratings produce output from -pi to pi
    #elif mask_type == 'unet':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spot_array[:, 1, 2] = spot_array[:, 0, 0]
    spot_array[:, 0, 0] = -spot_array[:, 0, 1]
    spot_array[:, 0, 1] = -spot_array[:, 1, 2]

    intensity_mask = input_converter.gen_input_intensity(spot_array)
    intensity_input = torch.tensor(intensity_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    phase_mask_og = model(intensity_input.to(device))
    phase_mask_u = phase_mask_og.detach().cpu().numpy().squeeze(0).squeeze(0)

    phase_mask = np.rot90(phase_mask, 1)
    phase_mask = np.flip(phase_mask, 0)

    phase_mask = ((phase_mask - phase_mask.min()) / (phase_mask.max() - phase_mask.min()) * 255).astype(np.uint8)
    phase_mask = cv2.resize(np.array(phase_mask), (512, 512), interpolation=0)

    phase_mask_u = ((phase_mask_u - phase_mask_u.min()) / (phase_mask_u.max() - phase_mask_u.min()) * 255).astype(np.uint8)
    phase_mask_u = cv2.resize(np.array(phase_mask_u), (512, 512), interpolation=0)

    phase_mask_u = np.rot90(phase_mask_u, 1)
    phase_mask_u = np.flip(phase_mask_u, 0)

    intensity_mask = ((intensity_mask - intensity_mask.min()) / (intensity_mask.max() - intensity_mask.min()) * 255).astype(np.uint8)
    intensity_mask = cv2.resize(np.array(intensity_mask), (512, 512))

    intensity_mask = np.rot90(intensity_mask, 1)
    intensity_mask = np.flip(intensity_mask, 0)


    def gaussian_beam(mask_size, beam_width):
        x = np.linspace(-1, 1, mask_size)
        y = np.linspace(-1, 1, mask_size)
        xx, yy = np.meshgrid(x, y)
        return np.exp(-(xx ** 2 + yy ** 2) / (2 * beam_width ** 2))

    def compute_beam_width(mask_sz, reference_size=512, reference_bm=0.05):
        return reference_bm * (reference_size / mask_sz)

    mask_size = 256
    bm = compute_beam_width(mask_size)
    incident_beam = gaussian_beam(mask_size, beam_width=bm)
    spot_array[:, 0, 0] = (spot_array[:, 0, 0]) * (49/120)
    spot_array[:, 0, 1] = (spot_array[:, 0, 1]) * (49/120)
    phase_mask_resized = jax.image.resize(calculate_phase_mask(spot_array, n // 16, 256, False)[0] * np.pi,
                                        (512, 512), 'nearest')

    phase_mask_resized = np.rot90(phase_mask_resized, 1)
    phase_mask_resized = np.flip(phase_mask_resized, 0)

    slm_field = gaussian_beam(mask_size, beam_width=bm) * np.exp(1j * phase_mask_resized)
    far_field = fftshift(fft2(slm_field)) / ((2 * mask_size) ** 2)
    intensity = np.abs(far_field) ** (1/2)
    intensity_farfield = intensity / np.max(intensity)
   #intensity_farfield = crop_upper_left_quadrant(crop_lower_right_quadrant(((intensity_farfield - intensity_farfield.min()) / (intensity_farfield.max() - intensity_farfield.min()) * 255).astype(np.uint8)))

    phase_mask_og = jax.image.resize(phase_mask_og.detach().cpu().numpy().squeeze(0).squeeze(0), (512, 512), 'nearest')
    slm_field_u = gaussian_beam(mask_size, beam_width=bm) * np.exp(1j * phase_mask_og)
    far_field_u = fftshift(fft2(slm_field_u)) / ((2 * mask_size) ** 2)
    intensity_u = np.abs(far_field_u) ** (1/2)
    intensity_farfield_u = intensity_u / np.max(intensity_u)
    #intensity_farfield_u = crop_upper_left_quadrant(crop_lower_right_quadrant(((intensity_farfield_u - intensity_farfield_u.min()) / (
    #            intensity_farfield_u.max() - intensity_farfield_u.min()) * 255).astype(np.uint8)))


    intensity_mask = cv2.resize(intensity_mask, (512, 512))
    intensity_farfield = cv2.resize(crop_upper_left_quadrant(intensity_farfield), (512, 512), interpolation=1)
    intensity_farfield_u = cv2.resize(crop_upper_left_quadrant(intensity_farfield_u), (512, 512), interpolation=1)

    window_width = 512
    window_height = 512

    # List of (title, image, position)
    windows = [
        ("Phase Mask", phase_mask, (0, 0)),
        ("Unwrapped Phase Mask", phase_mask_u, (window_width, 0)),
        ("Far Field", intensity_farfield, (window_width * 2, 0)),
        ("Unwrapped Far Field", intensity_farfield_u, (window_width * 3, 0)),
        ("Target Mask", intensity_mask, (window_width * 4, 0)),
    ]

    for title, image, pos in windows:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, window_width, window_height)
        cv2.moveWindow(title, *pos)
        cv2.imshow(title, image)

    cv2.waitKey(1)  # Use waitKey(0) to pause until keypress
    # combined = np.hstack([phase_mask, phase_mask_u, intensity_farfield, intensity_farfield_u, intensity_mask])
    # cv2.namedWindow("Mask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("Mask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.moveWindow("Mask", 0, 0)
    # cv2.resizeWindow("Mask", 2560, 512)
    #
    #
    # cv2.imshow("Mask", combined)
    # cv2.waitKey(1)

def crop_lower_right_quadrant(image, final_size=512):
    h, w = image.shape
    cropped = image[h//2:, w//2:]  # Lower right quadrant
    return cv2.resize(cropped, (final_size, final_size), interpolation=cv2.INTER_NEAREST)

def crop_upper_left_quadrant(image, final_size=512):
    h, w = image.shape
    cropped = image[:h//2, :w//2]  # Lower right quadrant
    return cv2.resize(cropped, (final_size, final_size), interpolation=cv2.INTER_NEAREST)

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