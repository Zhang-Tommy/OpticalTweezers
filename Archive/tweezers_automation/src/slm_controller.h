#pragma once

#pragma comment(lib, "ws2_32.lib")

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
#include <thread>
#include <mutex>

#include "spot.h"
#include "bead_detector.h"
#include "ExampleUtils.h"
#include "read_cam.h"

#define PORT 8080
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define COUT(string) cout<< string << endl

using namespace std;

Spot spots[100];
int num_spots = 0;
SOCKET udp_socket;
std::mutex m;
cv::Mat cam_img;

// Function declarations
char* read_file(string filePath);
void initialize_holo_engine();
int send_message(char* message);
int update_uniform(int uniform_var, float values[], int num_values);
int create_spot(float* spot_data);
int modify_spot(float* spot_data, int spot_index);
void random_spots_test();
float* get_spots();
