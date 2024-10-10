import os
import socket
import subprocess
import time
import numpy as np
import cv2

from harvesters.core import Harvester
from constants import *
from multiprocessing.managers import BaseManager

class SpotManagerManager(BaseManager):
    pass

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
        # if np.linalg.norm(np.array([x, y]) - np.array(line_goal[0])) < 15:
        #         #     line_goal.clear()
        #         #     #spot_man.remove_trap((trap[0], trap[1]))
        #         #     #traps.pop(j)
        #         #     return
        #         # if np.linalg.norm(np.array([x, y]) - np.array(donut_goal[0])) < 15:
        #         #     donut_goal.clear()
        #         #     #spot_man.remove_trap((trap[0], trap[1]))
        #         #     #traps.pop(j)
        #         #     return
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
        # add goal point
        spot_man.add_goal_pos((x,y))

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

def draw_traps(spot_man, frame, sim_man, donut_goal, line_goal):
    """
    Draws line, donut, and point traps on the displayed image
    """
    traps = spot_man.get_trapped_beads()
    goals = spot_man.get_goal_pos()

    if donut_goal:
        cv2.circle(frame, (donut_goal[0][0], donut_goal[0][1]), 4, (0, 128, 256), -1)
        cv2.circle(frame, (donut_goal[0][0], donut_goal[0][1]), 8, (0, 128, 256), 1)
    if line_goal:
        cv2.circle(frame, (line_goal[0][0], line_goal[0][1]), 4, (64, 128, 0), 1)

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
            angle = spot.angle

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