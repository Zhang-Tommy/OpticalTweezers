// SLMController.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <winsock2.h>
#include <iostream>
#include <windows.h>
#include <fstream>
#include <string>
#include <WS2tcpip.h>
#include <cstring>
#include <chrono>
#include <thread>
#include <shellapi.h>
#include <format>
#include "spot.h"
#include "bead_detector.h"

#pragma comment(lib, "ws2_32.lib")

#define PORT 8080
#define _WINSOCK_DEPRECATED_NO_WARNINGS

using namespace std;

Spot spots[100];
int num_spots = 0;
SOCKET udp_socket;

// Function declarations
char* read_file(string filePath);
void initialize_holo_engine();
int send_message(char* message);
int update_uniform(int uniform_var, float values[], int num_values);
int create_spot(float* spot_data);
int modify_spot(float* spot_data, int spot_index);
void random_spots_test();
float* get_spots();

// Inputs: file path to the file being read
// Returns: a character array of the message
char* read_file(string filePath) {
    using namespace std::this_thread;
    using namespace std::chrono_literals;
    string line;
    ifstream myfile(filePath);

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


    /*
    const int BufLen = 10240;
    char RecvBuf[BufLen];

    int bind_addrSize = sizeof(bind_addr);

    int bytesReceived = recvfrom(udp_socket, RecvBuf, BufLen, 0, (sockaddr*)&bind_addr, &bind_addrSize);

    cout << bytesReceived;  // print out the bytes we receive from hologram engine
    
    if (bytesReceived == SOCKET_ERROR) {
        std::cerr << "recvfrom failed." << std::endl;
    }
    */

    sockaddr_in recv_addr;
    unsigned short Port = 61557;
    recv_addr.sin_family = AF_INET;
    recv_addr.sin_port = htons(Port);
    inet_pton(AF_INET, "127.0.0.1", &recv_addr.sin_addr);

    // Read shader source and uniform vars into variables
    char* shader_source = read_file("./shader_source.txt");
    char* init_uniform_vars = read_file("./init_uniform_vars.txt");

    // Send code and variables to hologram engine
    int result_shader = sendto(udp_socket, shader_source, 8425, 0, (SOCKADDR*)&recv_addr, sizeof(recv_addr)); // magic number is bytes of array
    int result_init = sendto(udp_socket, init_uniform_vars, 1408, 0, (SOCKADDR*)&recv_addr, sizeof(recv_addr));
   
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
int update_uniform(int uniform_var, float values[], int num_values) {
    sockaddr_in recv_addr;
    unsigned short Port = 61557;
    recv_addr.sin_family = AF_INET;
    recv_addr.sin_port = htons(Port);
    inet_pton(AF_INET, "127.0.0.1", &recv_addr.sin_addr);

    string packet = format("<data>\n<uniform id = {}>\n", uniform_var);

    for (int i = 0; i < num_values; i++) {
        packet += to_string(values[i]) + " ";
    }
    packet += "\n</uniform>\n</data>";

    int result_init = sendto(udp_socket, &packet[0], size(packet), 0, (SOCKADDR*)&recv_addr, sizeof(recv_addr));

    return result_init;
}

// Returns pointer to an array holding all the spot parameters
float* get_spots() {
    float* spot_vals = new float[num_spots * 16];
    int count = 0;
    // for every spot in the spot array, add the parameters into this array
    for (int i = 0; i < num_spots; i++) {
        for (int j = 0; j < 16; j++) {
            spot_vals[count] = spots[i].vals[j];
            count += 1;
        }
    }

    return spot_vals;
}

// Create a new spot and send over parameters to the hologram engine
// Inputs: float array of spot data/parameters
// Returns: number of bytes sent to hologram engine
int create_spot(float* spot_data) {
    Spot *new_spot = new Spot(spot_data);
    spots[num_spots + 1] = *new_spot;
    num_spots += 1;
    float* spot_vals = get_spots();
    int update_code = update_uniform(2, spot_vals, sizeof(float) * num_spots * 4);
    return update_code;
}

// Modifies the specified spot and sends updated uniform variable to hologram engine
// Inputs: Spot index, float array of spot data/parameters
// Returns: number of bytes sent to hologram engine
int modify_spot(float* spot_data, int spot_index) {
    spots[spot_index].set_spot_values(spot_data);
    float* spot_vals = get_spots();
    int update_code = update_uniform(2, spot_vals, sizeof(float) * num_spots * 4);
    return update_code;
}

// Randomly creates and modifies 50 spots to test the connection and hologram engine
void random_spots_test() {
    srand(static_cast<unsigned>(time(0)));

    // Create 50 random spots
    for (int i = 0; i < 50; i++) {
        float random_spot_data[16];
        for (int j = 0; j < 16; j++) {
            random_spot_data[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 5.0; 
        }

        create_spot(random_spot_data); 
        this_thread::sleep_for(chrono::milliseconds(100));
    }

    // Modify 50 random spots
    for (int i = 0; i < 50; i++) {
        int spot_index = rand() % num_spots; 
        float random_spot_data[16];
        for (int j = 0; j < 16; j++) {
            random_spot_data[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 5.0; 
        }

        modify_spot(random_spot_data, spot_index); 
        this_thread::sleep_for(chrono::milliseconds(100));
    }
}

int main()
{

    initialize_holo_engine(); // bind to the udp socket and intialize shader code
    
    random_spots_test(); // test the connection by creating and modifying random traps

    // create spots / line traps using the existing GPU functions
    // 
    // move spots in a linear direction with direction and speed
}

