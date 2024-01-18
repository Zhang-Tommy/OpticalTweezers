// tweezers_automation_v2.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#pragma comment(lib, "ws2_32.lib")
//#include <iostream>
#include "camera.h"
#include "spot_manager.h"
#include "udp_sender.h"
#include "pid.h"

#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <ctime>

std::mutex m; // mutex for cam_img variable
std::mutex k; // mutex for keypoints variable
std::mutex g;
std::vector<cv::KeyPoint> keypoints; // stores detected bead coords, continuously updated
std::vector<cv::KeyPoint> trap_points;
cv::Mat cam_img; // camera frame, continuously updated
bool terminate_all = false;

//SpotManager* spotManager;

std::pair<int, int> calculateCoordinates(double angle, int radius);
void test_bead_trajectory();
void test_bead_movement(SpotManager* spotManager);
void bead_tracking(SpotManager* spotManager);
void trap_bead_single();
void test_spot_manager();


int main()
{
    SpotManager* spotManager = new SpotManager();

    std::thread imaging(get_img_offline_test);
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    
    std::thread detecting(detect_beads);
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    
    //std::thread tracking(bead_tracking, spotManager);
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    test_bead_movement(spotManager);

    std::this_thread::sleep_for(std::chrono::milliseconds(100000));
    
    while (true) {
        int key = cv::waitKey(1);
        if (key == 27) {  // ASCII code for 'Esc'
            terminate_all = true;
            break;
        }
        
    }
    
}


// Identifies and uniquely labels all detected beads between consecutive frames
// Used for PID control of bead position
void bead_tracking(SpotManager* spotManager) {
    while (true) {
        if (terminate_all) {
            break;
        }
        int num_tracked = 0;
        {
            //std::lock_guard<std::mutex> lock_g(g);
            //std::lock_guard<std::mutex> lock_k(k);

            for (auto& bead : keypoints) {
                int x_detect = bead.pt.x;
                int y_detect = bead.pt.y;
                int radius = 6; // pixels
                int x_min = std::max(0, x_detect - radius);
                int x_max = std::min(480, x_detect + radius);
                int y_min = std::max(0, y_detect - radius);
                int y_max = std::min(640, y_detect + radius);

                for (int i = x_min; i < x_max; i++) {
                    for (int j = y_min; j < y_max; j++) {
                        // If we find a matching spot in the grid, 
                        if (spotManager->grid[j][i].assigned) {
                            

                            std::cout << "Assigned bead at: (" << i << ", " << j << ") ";
                            std::cout << "Detected bead at: (" << x_detect << ", " << y_detect << ")" << std::endl;
                            num_tracked += 1;


                            // PID is a class 
                            // We need two PID classes per bead
                            // Each bead that is eligible for PID control must be detected by keypoints and be in the grid
                            // That is already true since we are in this loop
                            double control_x = spotManager->grid[j][i].pid_x->calculate(i, x_detect);
                            double control_y = spotManager->grid[j][i].pid_y->calculate(j, y_detect);
                            std::cout << "PID output: (" << control_x + i << ", " << control_y + j << ")" << std::endl;

                            // With the control_x and control_y values we can update the grid spot parameter and send to holo engine

                            spotManager->grid[j][i].set_new_pos(control_x, control_y);
                            spotManager->update_traps();

                        }
                    }
                }
            }
            
        }
        std::cout << keypoints.size() << " ";
        std::cout << num_tracked << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}


// Automatic bead movement test
// Trap all detected beads, then translate them to the right edge of the work area
void test_bead_movement(SpotManager* spotManager) {

    {
        std::lock_guard<std::mutex> lock_g(g);
        std::lock_guard<std::mutex> lock_k(k);
        for (auto& keypoint : keypoints) {
            spotManager->create_trap(keypoint.pt.y, keypoint.pt.x);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    int pxls_moved = 0;
    
    std::list<std::pair<int, int>> beadsToMove;

    for (const auto& bead : spotManager->trapped_beads) {
        beadsToMove.push_back(bead.first);
    }

    for (const auto& bead : beadsToMove) {
        spotManager->translate_trap(bead.first, bead.second, 300, 300, 2);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }


    /*
    while (true) {
        if (terminate_all) {
            break;
        }
        if (pxls_moved == 300) {
            break;
        }

        std::list<std::pair<int, int>> beadsToMove;

        // Populate the list with beads to move
        for (const auto& bead : spotManager->trapped_beads) {
            beadsToMove.push_back(bead.first);
        }

        
        // Move beads
        {
            std::lock_guard<std::mutex> lock_g(g);
            for (const auto& bead : beadsToMove) {
                spotManager->move_trap(bead.first, bead.second, bead.first, bead.second);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        

        pxls_moved += 1;
    }
   */
    // how to do PID control?
    // modify grid[][] spot variable parameters then update_spots();
    // new keypoints will be coming in, these will be used to determine error
    // only problem is that these keypoints are not used to identify a bead
    // how to ensure tracking of a specific bead?
}

void trap_bead_single() {
    SpotManager* spotManager = new SpotManager();
    std::cout << "Wait for holoengine" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    int counter = 0;

    for (auto& keypoint : keypoints) {
        spotManager->create_trap(keypoint.pt.y, keypoint.pt.x);
        counter += 1;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::cout << "Trapped " << counter << " beads" << std::endl;
}

// Function to calculate the coordinates of a point on a circle given an angle
std::pair<int, int> calculateCoordinates(double angle, int radius) {
    int centerX = 240;
    int centerY = 320;
    int x = static_cast<int>(radius * std::cos(angle)) + centerX;
    int y = static_cast<int>(radius * std::sin(angle)) + centerY;
    return std::make_pair(y, x);
}

// Test a desired trajectory mainly to validate PID control of bead positioning.
// Also use for ensuring calibration of tweezers setup and software scaling factors
void test_bead_trajectory() {
    SpotManager* spotManager = new SpotManager();

    std::thread tracking(bead_tracking, spotManager);

    std::vector<std::pair<int, int>> trajectory;
    // Move a single bead to starting position



    // Define the parameters for the circular trajectory
    int radius = 75;
    int numWaypoints = 360; // Number of waypoints to complete a full circle

    int counter = 0;
    // Calculate waypoints for the circular trajectory
    for (int angle = 0; angle < numWaypoints; ++angle) {
        double radians = angle * (3.14 / 180.0); // Convert degrees to radians
        auto coordinates = calculateCoordinates(radians, radius);
        trajectory.push_back(coordinates);
    }

    int start_x = trajectory.front().second;
    int start_y = trajectory.front().first;

    std::cout << "created trap at " << start_x << ", " << start_y << std::endl;
    spotManager->create_trap(start_y, start_x);
    std::cout << "waiting to ensure bead is trapped" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    std::cout << "done waiting for bead to be trapped" << std::endl;
    // Print the waypoints
    int prev_x = start_x;
    int prev_y = start_y;
    for (const auto& point : trajectory) {
        if (counter == 0) {
            counter += 1;
        }
        else {
            std::cout << "(" << point.first << ", " << point.second << ")" << std::endl;
            spotManager->move_trap(prev_y, prev_x, point.first, point.second);
            prev_y = point.first;
            prev_x = point.second;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

    }
    std::cout << "Done moving bead in circular trajectory" << std::endl;
}



// Stress test the spot manager class
void test_spot_manager() {
    //get_img_offline_test();
    // Seed for random number generation
    //std::srand(std::time(0));

    // Create an instance of SpotManager
    SpotManager* spotManager = new SpotManager();

    // Number of operations to perform
    const int numOperations = 100;


    for (int i = 0; i < numOperations; ++i) {
        // Generate random coordinates within the range [0, 400]
        int x = std::rand() % 401;
        int y = std::rand() % 401;

        // Generate random values within the range [0, 1]
        float z = static_cast<float>(std::rand()) / RAND_MAX;
        int l = std::rand() % 2; // Random integer 0 or 1
        float intensity = static_cast<float>(std::rand()) / RAND_MAX;
        float phase = static_cast<float>(std::rand()) / RAND_MAX;

        // Perform a random operation: create, move, or remove trap
        int operation = std::rand() % 3;

        if (operation == 0) {
            // Create trap
            spotManager->create_trap(x, y, static_cast<int>(z * 100), l, intensity, phase);
            std::cout << "Created trap at (" << x << ", " << y << ")" << std::endl;
        }
        else if (operation == 1) {
            // Check if there is a trap at the specified location before moving
            if (spotManager->grid[x][y].assigned) {
                int newX = std::rand() % 401;
                int newY = std::rand() % 401;
                spotManager->move_trap(x, y, newX, newY);
                std::cout << "Moved trap from (" << x << ", " << y << ") to (" << newX << ", " << newY << ")" << std::endl;
            }
            else {
                std::cout << "No trap to move at (" << x << ", " << y << ")" << std::endl;
            }
        }
        else {
            // Check if there is a trap at the specified location before removing
            if (spotManager->grid[x][y].assigned) {
                spotManager->remove_trap(x, y);
                std::cout << "Removed trap at (" << x << ", " << y << ")" << std::endl;
            }
            else {
                std::cout << "No trap to remove at (" << x << ", " << y << ")" << std::endl;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
}