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
std::vector<cv::KeyPoint> keypoints; // stores detected bead coords, continuously updated
cv::Mat cam_img; // camera frame, continuously updated

void test_bead_movement();
void bead_tracking(SpotManager* spotManager);

int main()
{
    std::thread imaging(get_img_offline_test);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    std::thread detecting(detect_beads);
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    
    test_bead_movement();

    return 0;
}

// Identifies and uniquely labels all detected beads between consecutive frames
// Used for PID control of bead position
void bead_tracking(SpotManager* spotManager) {
    // Need to find the error of commanded vs. actual position.
    // The commanded position is stored in the grid spot array
    // The actual position is found through bead detection
    // Need to link/associate each detected bead with a grid bead.
    // Needs to be robust since beads can wander on and off the work space
    
    // Essentially map keypoints to grid
    // Maybe can search the grid based on the keypoints?
    
    // One pixel is roughly 0.1875 um
    // If the update rate of bead tracking is 5 hz? What is reasonable threshold for the brownian motion of the bead?
    // Lets say 1 um or 6 pixels

    // for (bead in keypoints) 
    // check coordinate in grid
    // if any beads are assigned within search range then we assume the actual position is from keypoint
    std::ofstream csvFile("keypoints.csv");


    while (true) {
        //std::lock_guard<std::mutex> lock_k(k);
        int num_tracked = 0;
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        struct tm* tm_gmt = std::gmtime(&now);

        csvFile << std::put_time(tm_gmt, "%Y-%m-%d %H:%M:%S") << ",";

        auto startTime = std::chrono::high_resolution_clock::now();
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
                        csvFile << x_detect << "," << y_detect << "," << i << "," << j << ",";

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
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        //std::cout << duration << std::endl;

        csvFile << std::endl;
        std::cout << keypoints.size() << " ";
        std::cout << num_tracked << std::endl;

        int key = cv::waitKey(300);

        if (key == 27) {
            std::cout << "Break" << std::endl;
            break;
        }

        //std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    csvFile.close();

    // Like in the beginning we can trap the beads we want to move
    // then with those trapped beads we can track them by imagining we place it onto the grid corresponding to it's coordinates. 
    // Look around where we placed the keypoint. If there is an assigned spot nearby then we know that one is the one that the keypoint corresponds to.

    // PID control
    // Can modify the grid[][] spot and update to hologram engine
    // The desired coordinates are still stored and unchanged in trapped_beads and the index to grid[][]
    // Can maybe run this in a thread, or else just do this every few moves
    //
}




// Automatic bead movement test
// Trap all detected beads, then translate them to the right edge of the work area
void test_bead_movement() {
    SpotManager* spotManager = new SpotManager();

    for (auto& keypoint : keypoints) {
        spotManager->create_trap(keypoint.pt.y, keypoint.pt.x);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    // for bead in trapped_beads
    // move bead from old to new location
    /*
    while (true) {
        for (auto& bead : spotManager->trapped_beads) {
            spotManager->move_trap(bead.first.first, bead.first.second, bead.first.first, bead.first.second + 1);
        }
    }
    */
    int pxls_moved = 0;
    std::thread tracking(bead_tracking, spotManager);

    while (true) {
        if (pxls_moved == 200) {
            break;
        }

        std::list<std::pair<int, int>> beadsToMove;

        // Populate the list with beads to move
        for (const auto& bead : spotManager->trapped_beads) {
            beadsToMove.push_back(bead.first);
        }

        // Move beads
        for (const auto& bead : beadsToMove) {
            spotManager->move_trap(bead.first, bead.second, bead.first, bead.second);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        pxls_moved += 1;
    }

    // how to do PID control?
    // modify grid[][] spot variable parameters then update_spots();
    // new keypoints will be coming in, these will be used to determine error
    // only problem is that these keypoints are not used to identify a bead
    // how to ensure tracking of a specific bead?
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