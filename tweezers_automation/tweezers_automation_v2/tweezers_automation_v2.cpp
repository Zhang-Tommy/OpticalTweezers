// tweezers_automation_v2.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#pragma comment(lib, "ws2_32.lib")
#include "camera.h"
#include "spot_manager.h"
#include "udp_sender.h"
#include "pid.h"
#include "motion_planner.h"

#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <ctime>

std::mutex m; // mutex for cam_img
std::mutex k; // mutex for keypoints
std::mutex g; // mutex for grid
std::vector<cv::KeyPoint> keypoints; // stores detected bead coords, continuously updated
cv::Mat cam_img; // camera frame, continuously updated
std::vector<cv::KeyPoint> usr_points;

std::pair<int, int> calculate_coordinates(double angle, int radius);
void test_bead_trajectory();
void test_bead_movement(SpotManager* spotManager);
void test_motion_planner(SpotManager* spotManager);
void test_dual_motion(SpotManager* spotManager);
void follow_path(SpotManager* spotManager, std::vector<std::pair<int, int>> path, int start_x, int start_y);
//void addBufferRegion(std::vector<cv::KeyPoint>& obstacles, int radius);


int main()
{
    SpotManager* spotManager = new SpotManager();

    std::thread imaging(get_img_offline_test, spotManager);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    std::thread detecting(detect_beads);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    std::thread tracking(bead_tracking, spotManager);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    //test_dual_motion(spotManager);
    //test_motion_planner(spotManager);
    //test_bead_movement(spotManager);

    std::this_thread::sleep_for(std::chrono::milliseconds(100000));
    
}


void single_bead_stability(SpotManager* spotManager, int um_sec) {
    {
        std::lock_guard<std::mutex> lock_g(g);
        std::lock_guard<std::mutex> lock_k(k);
        int count = 0;

        for (auto& keypoint : keypoints) {
            if (count == 1) {
                break;
            }
            spotManager->create_trap(keypoint.pt.y, keypoint.pt.x);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            count += 1;
        }
    }

    std::list<std::pair<int, int>> beads_to_move;

    for (const auto& bead : spotManager->trapped_beads) {
        beads_to_move.push_back(bead.first);
    }

    for (const auto& bead : beads_to_move) {
        spotManager->translate_trap(bead.first, bead.second, 240, 600, um_sec);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}


// Test motion planner for simultaneous motion of two beads
void test_dual_motion(SpotManager* spotManager) {
    float y_one = keypoints[1].pt.y;
    float x_one = keypoints[1].pt.x;
    float y_two = keypoints[2].pt.y;
    float x_two = keypoints[2].pt.x;

    spotManager->create_trap(y_one, x_one);
    spotManager->create_trap(y_two, x_two);

    std::pair<int, int> start_one = std::make_pair(x_one / (640 / GRID_X), y_one / (480 / GRID_Y));
    std::pair<int, int> start_two = std::make_pair(x_two / (640 / GRID_X), y_two / (480 / GRID_Y));

    std::pair<int, int> goal_one = std::make_pair(0, 0);
    std::pair<int, int> goal_two = std::make_pair(30, 15);

    Planner* plan_one = new Planner(spotManager);
    Planner* plan_two = new Planner(spotManager);

    plan_one->get_obstacles();
    plan_one->bfs(start_one, goal_one);

    std::vector<std::pair<int, int>> path_one = plan_one->backtrack(start_one, goal_one);

    // for every point in path_one, add as obstacle to path_two
    std::vector<cv::KeyPoint> path_one_obstacles;
    for (auto& point : path_one) {
        path_one_obstacles.push_back(cv::KeyPoint(point.first * (640 / GRID_X), point.second * (480 / GRID_Y), 20));
    }

    plan_two->add_obstacles(path_one_obstacles);
    plan_two->get_obstacles();
    plan_two->bfs(start_two, goal_two);

    std::vector<std::pair<int, int>> path_two = plan_two->backtrack(start_two, goal_two);

    //follow_path(spotManager, path_one, x_one, y_one);
    std::thread execute_p1(follow_path, spotManager, path_one, x_one, y_one);
    std::thread execute_p2(follow_path, spotManager, path_two, x_two, y_two);
    std::this_thread::sleep_for(std::chrono::milliseconds(100000));
}


void follow_path(SpotManager* spotManager, std::vector<std::pair<int, int>> path, int start_x, int start_y) {
    
    int prev_x = start_x;
    int prev_y = start_y;
    
    path.erase(path.begin());

    for (auto& elem : path) {
        // scale element back to camera coords
        int resize_x = static_cast<float>(elem.first) * (640.0 / GRID_X);
        int resize_y = static_cast<float>(elem.second) * (480.0 / GRID_Y);
        //printf("%i, %i", resize_x, resize_y);
        //g.lock();
        //spotManager->translate_trap(prev_y, prev_x, resize_y, resize_x, 20);
        spotManager->move_trap(prev_y, prev_x, resize_y, resize_x);
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        //g.unlock();
        prev_y = resize_y;
        prev_x = resize_x;
        //std::cout << "(" << prev_y << ", " << prev_x << ")\n";
    }
    
}


// Motion planner test
// Trap a single bead, then move it to another location
// Use the motion planner coordinate vector for path
void test_motion_planner(SpotManager* spotManager) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    float y_keypoint = keypoints[1].pt.y;
    float x_keypoint = keypoints[1].pt.x;
    spotManager->create_trap(y_keypoint, x_keypoint);

    Planner* plan = new Planner(spotManager);
    plan->get_obstacles();
    //plan->add_obstacles(obstacles);
    
    std::pair<int, int> start = std::make_pair(x_keypoint / (640 / GRID_X), y_keypoint / (480 / GRID_Y)); // this should be the bead we want to move
    std::pair<int, int> goal = std::make_pair(1, 39);

    plan->bfs(start, goal);

    std::vector<std::pair<int, int>> path = plan->backtrack(start, goal);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;

    int prev_x = x_keypoint;
    int prev_y = y_keypoint;

    for (auto& elem : path) {
        // scale element back to camera coords
        int resize_x = static_cast<float>(elem.first) * (640.0 / GRID_X);
        int resize_y = static_cast<float>(elem.second) * (480.0 / GRID_Y);
        //printf("%i, %i", resize_x, resize_y);
        spotManager->translate_trap(prev_y, prev_x, resize_y, resize_x, 20);
        //std::this_thread::sleep_for(std::chrono::milliseconds(250));

        prev_y = resize_y;
        prev_x = resize_x;
        //std::cout << "(" << prev_y << ", " << prev_x << ")\n";
    }

    // for every element in path, move the bead in desired trajectory
}


// Automatic bead movement test
// Trap all detected beads, then translate them to the right edge of the work area
void test_bead_movement(SpotManager* spotManager) {
    {
        std::lock_guard<std::mutex> lock_g(g);
        std::lock_guard<std::mutex> lock_k(k);
        int count = 0;


        for (auto& keypoint : keypoints) {
            if (count == 5) {
                break;
            }
            spotManager->create_trap(keypoint.pt.y, keypoint.pt.x);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            count += 1;
        }
    }
    
    
    std::list<std::pair<int, int>> beads_to_move;

    for (const auto& bead : spotManager->trapped_beads) {
        beads_to_move.push_back(bead.first);
    }

    for (const auto& bead : beads_to_move) {
        spotManager->translate_trap(bead.first, bead.second, 400, 600, 10);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
}


// Function to calculate the coordinates of a point on a circle given an angle
std::pair<int, int> calculate_coordinates(double angle, int radius) {
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
        auto coordinates = calculate_coordinates(radians, radius);
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
