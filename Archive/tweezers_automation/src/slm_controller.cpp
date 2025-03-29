// SLMController.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#pragma comment(lib, "ws2_32.lib")

#include <winsock2.h>
#include <iostream>
#include <fstream>

#include "Serial.h"
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
//#include "bgapi2_genicam.hpp"
#include "spot.h"
#include "camera_utils.h"



//using namespace std;
//using namespace std::chrono;

#define PORT 8080
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define COUT(string) cout<< string << endl
//#define CAM_FRAME_RATE 100

// Global variables
Spot spots[100];
int num_spots = 0;
SOCKET udp_socket;
std::mutex m;
std::mutex k;
cv::Mat cam_img;
std::vector<cv::KeyPoint> keypoints;

// Function declarations
char* read_file(std::string filePath);
void initialize_holo_engine();
int send_message(char* message);
int update_uniform(int uniform_var, float values[], int num_values);
int create_spot(float* spot_data);
int modify_spot(float* spot_data, int spot_index);
void random_spots_test();
float* get_spots();
//int get_img();
//int detect_beads();


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
    size_t shader_source_length = strlen(shader_source);
    size_t init_length = strlen(init_uniform_vars);
    // Send code and variables to hologram engine
    int result_shader = sendto(udp_socket, shader_source, shader_source_length, 0, (SOCKADDR*)&recv_addr, sizeof(recv_addr)); // magic number is bytes of array
    int result_init = sendto(udp_socket, init_uniform_vars, init_length, 0, (SOCKADDR*)&recv_addr, sizeof(recv_addr));

    //if (result_shader < 0 || result_init < 0) {
    //    std::cerr << "shader or init code failed to send" << std::endl;
    //}

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

    std::string packet = std::format("<data>\n<uniform id = {}>\n", uniform_var);

    for (int i = 0; i < num_values; i++) {
        packet += std::to_string(values[i]) + " ";
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
    // Todo: implement a check for bounds in physical space (120x90um)
    Spot* new_spot = new Spot(spot_data);
    num_spots += 1;
    if (num_spots == 0) {
        spots[0] = *new_spot;
    }
    else {
        spots[num_spots] = *new_spot;

    }

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
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Modify 50 random spots
    for (int i = 0; i < 50; i++) {
        int spot_index = rand() % num_spots;
        float random_spot_data[16];
        for (int j = 0; j < 16; j++) {
            random_spot_data[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 5.0;
        }

        modify_spot(random_spot_data, spot_index);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Creates a trap at a specified point and moves it horizontally at a specified velocity
// Inputs: x and y are initial coords of trap (in micrometers (um)), um_sec um/sec 
// um_distance movement distance
void line_path(float y, float x, int um_sec, int um_distance) {
    int slm_refresh = 100; // refresh rate of slm (fastest we can update the hologram)
    float spot_params[16] = { y, -x, 0.0, 0.0,
                            1.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 1.0,
                            0.0, 0.0, 0.0, 0.0 };
    create_spot(spot_params);
    // pxls / second
    // for loop runs 10 times per second
    int time = 0;
    int pxl = x;
    while (true) {
        // element 0 x  y  z  l    (x,y,z in um and l is an integer)
        // element 1 intensity (I) phase -  -
        // element 2 na.x na.y na.r -  (the x, y, and radius, of the spot on the SLM- useful for Shack-Hartmann holograms)
        // element 3 line trapping x y z and phase gradient.  xyz define the size and angle of the line, phase gradient (between +/-1) is the
        // scattering force component along the line.  Zero is usually a good choice for in-plane line traps
        float n_spot_params[16] = { y, -pxl, 0.0, 0.0,
                                    1.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 1.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0 };
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / um_sec));
        modify_spot(n_spot_params, 0);
        pxl += 1.0;
        //COUT("pxl_move");
        if (pxl == um_distance + x) {
            break;
        }
    }

}

void testing_line_path() {
    initialize_holo_engine(); // bind to the udp socket and intialize shader code

    line_path(70.0, 0.0, 10, 120);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    line_path(0.0, 0.0, 10, 20);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    line_path(35.0, 0.0, 10, 120);
}

void test_serial() {
    tstring commPortName(TEXT("COM3"));
    Serial serial(commPortName, 57600);
    char set_pow[] = "SDC 35";
    int bytesWritten = serial.write(set_pow);
    std::cout << std::format("{} bytes written to serial port", bytesWritten) << std::endl;
    char buffer[20];

    std::cout << "Reading from the serial port: ";
    for (int i = 0; i < 10; i++)
    {
        int charsRead = serial.read(buffer, 20);
        std::cout << buffer;
        Sleep(100);
    }
    std::cout << std::endl;

}

void test_cam_detect() {
    k.lock();
    for (int i = 0; i < keypoints.size(); i++) {
        std::cout << "start trapping all beads";
        float spot_params[16] = { keypoints[i].pt.y * 0.1875, -keypoints[i].pt.x * 0.1875, 0.0, 0.0,
                                1.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 1.0,
                                0.0, 0.0, 0.0, 0.0 };
        create_spot(spot_params);
    }
    k.unlock();

    int move_dist = 100;  // move 100um

    // Move all beads to right edge of screen
    int um_sec = 1;
    for (int i = 0; i < move_dist; i++) {
        //m.lock();
        //detect_beads();
        //m.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / um_sec));
        k.lock();
        for (int j = 0; j < keypoints.size(); j++) {
            if (keypoints[j].pt.x < 640) {
                std::cout << "Keypoints y, x:" << " ";
                std::cout << keypoints[j].pt.y << ", ";
                std::cout << keypoints[j].pt.x << std::endl;

                float n_spot_params[16] = { keypoints[j].pt.y * 0.1875,  (-keypoints[j].pt.x * 0.1875) - 1, 0.0, 0.0,
                                        1.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 1.0,
                                        0.0, 0.0, 0.0, 0.0 };
                modify_spot(n_spot_params, j);
            }
        }
        k.unlock();
    }

}

int main()
{
    initialize_holo_engine();
    std::thread imaging(get_img);
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    std::thread detecting(detect_beads);
    test_cam_detect();
}
