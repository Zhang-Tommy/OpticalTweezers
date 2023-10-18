import socket, os, subprocess, time, random, numpy as np
from utilities import *
from spot import Spot

ZERO_SPOTS = True
SPOTS_VEC = np.zeros(3200)
SPOTS = []
NUM_SPOTS = 1


def init_holo_engine():
    """Initializes hologram engine by sending shader source code and updating uniform variables
    Communicates with the hologram engine via UDP packets
    """

    # Kill any instances of hologram engine
    os.system("taskkill /f /im  hologram_engine_64.exe")
    subprocess.Popen([r'C:\Users\Tommy\PycharmProjects\OpticalTweezers\red_tweezers_1_4_release\hologram_engine_64.exe'])

    time.sleep(1)
    with open('shader_source.txt', 'r') as file:
        shader_source = file.read()
    with open('init_uniform_vars.txt', 'r') as file:
        uniform_vars = file.read()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(('127.0.0.1', 61556))

    server_socket.sendto(str.encode(shader_source), ('127.0.0.1', 61557))
    server_socket.sendto(str.encode(uniform_vars), ('127.0.0.1', 61557))


def add_spot(pos, intensity, phase):
    """Creates a new spot and sends it over to the hologram engine for rendering"""
    start = '<uniform id=2>\n'
    end = '\n</uniform>'

    # add the desired spot to the spot object
    new_spot = Spot()
    new_spot.change_pos(pos)
    new_spot.change_intensity(intensity)
    new_spot.change_phase(phase)

    global NUM_SPOTS

    for i in range(NUM_SPOTS * 16 - 16, NUM_SPOTS * 16):
        SPOTS_VEC[i] = new_spot.get_spot_params()[i % 16]

    # Format the numbers with 6 decimal places and join them with a single space
    string = ' '.join(f'{val:.6f}' for val in SPOTS_VEC[0:NUM_SPOTS * 16])

    packet = start + string + end

    send_data(packet)
    NUM_SPOTS += 1
    SPOTS.append(new_spot)

def randomize_spots():
    pos = (random.randrange(0, 7000), random.randrange(0, 7000), random.randrange(0, 7), 0)
    intensity = random.random()
    phase = random.random() * np.pi
    add_spot(pos, intensity, phase)


init_holo_engine()

randomize_spots()

spot = SPOTS[0]

for i in range(7000):
    spot.change_pos([i, i, 2, 0])
    update_spots(SPOTS_VEC, NUM_SPOTS, SPOTS)
    time.sleep(0.1)

kill_holo_engine()
