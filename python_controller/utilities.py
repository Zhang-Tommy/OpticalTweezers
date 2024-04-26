import socket, os
from constants import *
import subprocess, time
"""Functions for setting uniform variables"""

def init_holo_engine():
    """Initializes hologram engine by sending shader source code and updating uniform variables
    Communicates with the hologram engine via UDP packets
    """

    # Kill any instances of hologram engine
    os.system("taskkill /f /im  hologram_engine_64.exe")
    subprocess.Popen([r'.\hologram_engine_64.exe'])

    time.sleep(1)
    with open('shader_source.txt', 'r') as file:
        shader_source = file.read()
    with open('init_uniform_vars.txt', 'r') as file:
        uniform_vars = file.read()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(('127.0.0.1', 61556))

    server_socket.sendto(str.encode(shader_source), ('127.0.0.1', 61557))
    server_socket.sendto(str.encode(uniform_vars), ('127.0.0.1', 61557))

def send_data(message):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    start = '<data>\n'
    end = '\n</data>'
    server_socket.sendto(str.encode(start + message + end), (UDP_IP, UDP_PORT))


def set_uniform(var_num, new_data):
    start = f'<uniform id={var_num}>\n'
    end = '\n</uniform>'
    send_data(start + new_data + end)


def update_spots(SPOTS_VEC, NUM_SPOTS, SPOTS):
    start = '<uniform id=2>\n'
    end = '\n</uniform>'

    counter = 0
    for spot in SPOTS:
        for i in range(16):
            SPOTS_VEC[i] = spot.get_spot_params()[i]
        counter += 1

    # Format the numbers with 6 decimal places and join them with a single space
    string = ' '.join(f'{val:.6f}' for val in SPOTS_VEC[0:NUM_SPOTS * 16])

    packet = start + string + end
    #print(packet)
    send_data(packet)


# Converts from camera coordinates to micrometers in workspace
def cam_to_um(cam_pos):
    um_x = cam_pos[0] * CAM_TO_UM
    um_y = cam_pos[1] * CAM_TO_UM

    return [um_x, um_y]


def kill_holo_engine():
    """Terminate the hologram engine executable"""
    os.system("taskkill /f /im  hologram_engine_64.exe")