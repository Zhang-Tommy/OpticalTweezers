
#include <winsock2.h>
#include <iostream>
#include <fstream>

#include <string>
#include <windows.h>

#include <WS2tcpip.h>
#include <cstring>
#include <chrono>
#include <thread>
#include <shellapi.h>
#include <format>
#include <thread>
#include <mutex>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "udp_sender.h"

SOCKET udp_socket;


// Inputs: file path to the file being read
// Returns: a character array of the message
char* read_file(std::string filePath) {
    using namespace std::this_thread;
    using namespace std::chrono_literals;
    std::string line;
    std::ifstream myfile(filePath);

    if (myfile.is_open())
    {
        char* msg_char = new char[10001];
        int curr_index = 0;
        while (getline(myfile, line)) {
            for (int i = 0; i < line.length(); i++) {
                msg_char[curr_index] = line[i];
                curr_index += 1;
            }
            msg_char[curr_index] = '\n';
            curr_index += 1;
        }

        char* msg_string = msg_char;
        myfile.close();

        return msg_string;
    }
    return NULL;
}

// Connects to the hologram engine by binding to the udp port
// Sends shader code and initial uniform vars values to hologram engine
void initialize_holo_engine() {
    // Initialize winsock and socket
    WSAData wsa_data;
    WSAStartup(MAKEWORD(2, 2), &wsa_data);
    udp_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    sockaddr_in bind_addr;

    // Define ip address and port
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_port = 61556;
    inet_pton(AF_INET, "127.0.0.1", &bind_addr.sin_addr);

    sockaddr_in send_addr;
    send_addr.sin_family = AF_INET;
    send_addr.sin_port = 61557;
    inet_pton(AF_INET, "127.0.0.1", &send_addr.sin_addr);

    // Bind to the hologram engine
    bind(udp_socket, (sockaddr*)&bind_addr, sizeof(bind_addr));

    sockaddr_in recv_addr;
    unsigned short Port = 61557;
    recv_addr.sin_family = AF_INET;
    recv_addr.sin_port = htons(Port);
    inet_pton(AF_INET, "127.0.0.1", &recv_addr.sin_addr);

    // Read shader source and uniform vars into variables
    char* shader_source = read_file("./shader_source.txt");
    char* init_uniform_vars = read_file("./init_uniform_vars.txt");
    size_t shader_source_length = strlen(shader_source);
    size_t init_length = strlen(init_uniform_vars);
    // Send code and variables to hologram engine
    int result_shader = sendto(udp_socket, shader_source, shader_source_length, 0, (SOCKADDR*)&recv_addr, sizeof(recv_addr)); // magic number is bytes of array
    int result_init = sendto(udp_socket, init_uniform_vars, init_length, 0, (SOCKADDR*)&recv_addr, sizeof(recv_addr));

    if (result_shader < 0 || result_init < 0) {
       std::cerr << "shader or init code failed to send" << std::endl;
    }

    //closesocket(udp_socket);
    //WSACleanup();
}

int send_message(char* message) {
    sockaddr_in send_addr;
    send_addr.sin_family = AF_INET;
    send_addr.sin_port = 61557;
    inet_pton(AF_INET, "127.0.0.1", &send_addr.sin_addr);
    int result = sendto(udp_socket, message, sizeof(message), 0, (sockaddr*)&send_addr, sizeof(send_addr));

    return 0;
}

// Updates the specified uniform variable to desired value
int update_uniform(int uniform_var, const std::vector<float>& values, int num_values) {
    sockaddr_in recv_addr;
    unsigned short Port = 61557;
    recv_addr.sin_family = AF_INET;
    recv_addr.sin_port = htons(Port);
    inet_pton(AF_INET, "127.0.0.1", &recv_addr.sin_addr);

    std::string packet = std::format("<data>\n<uniform id = {}>\n", uniform_var);

    for (int i = 0; i < num_values; i++) {
        packet += std::to_string(values[i]) + " ";
    }
    //std::cout << packet;
    packet += "\n</uniform>\n</data>";

    int result_init = sendto(udp_socket, packet.c_str(), packet.size(), 0, (SOCKADDR*)&recv_addr, sizeof(recv_addr));

    return result_init;
}

