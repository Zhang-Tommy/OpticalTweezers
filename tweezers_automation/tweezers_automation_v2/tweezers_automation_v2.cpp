// tweezers_automation_v2.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#pragma comment(lib, "ws2_32.lib")
//#include <iostream>
#include "camera.h"
#include "spot_manager.h"
#include "udp_sender.h"

#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

std::mutex m;
std::mutex k;
std::vector<cv::KeyPoint> keypoints;
cv::Mat cam_img;

void test_bead_movement();

int main()
{
    std::thread imaging(get_img_offline_test);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    std::thread detecting(detect_beads);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    test_bead_movement();

    return 0;
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
            spotManager->move_trap(bead.first, bead.second, bead.first, bead.second + 1);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        pxls_moved += 1;
    }
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