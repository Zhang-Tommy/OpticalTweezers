import os
import socket
import subprocess
import time

from constants import *


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