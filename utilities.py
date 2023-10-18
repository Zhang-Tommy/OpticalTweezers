import socket, os

"""Functions for setting uniform variables"""
UDP_PORT = 61557
UDP_IP = '127.0.0.1'


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
    print(packet)
    send_data(packet)


def kill_holo_engine():
    """Terminate the hologram engine executable"""
    os.system("taskkill /f /im  hologram_engine_64.exe")