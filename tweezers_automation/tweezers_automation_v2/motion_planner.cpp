
#include "motion_planner.h"
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>



extern std::vector<cv::KeyPoint> keypoints;
extern std::mutex k;


Planner::Planner(SpotManager* spotManager) : spotManager(spotManager) {

}



void Planner::get_obstacles() {
    k.lock();
    // Find difference between currently tweezed beads and detected beads, these are obstacles
    std::vector<cv::KeyPoint> obstacles = keypoints;
    for (auto& point : obstacles) {
        obstacles.erase(std::remove_if(obstacles.begin(), obstacles.end(),
            [&](const cv::KeyPoint& keypoint) {
                int x = static_cast<int>(keypoint.pt.x);
                int y = static_cast<int>(keypoint.pt.y);
                return spotManager->trapped_beads.count(std::make_pair(y, x));
            }),
            obstacles.end());
    }

    // for every obstacle
    // for int i = obstacle.x, i < obstacle.x + radius
    // for int j = obstacle.y, j < obstacle.y + radius
    // if in the grid
    // set as obstacle
    int radius = 2;
    for (auto& obstacle : obstacles) {
        int obs_x = obstacle.pt.x / (640 / GRID_X);
        int obs_y = obstacle.pt.y / (480 / GRID_Y);
        for (int i = obs_x - radius; i < obs_x + radius + 1; i++) {
            for (int j = obs_y - radius; j < obs_y + radius + 1; j++) {
                if (!(i < 0 || i > GRID_X || j < 0 || j > GRID_Y)) {
                    obstacles.push_back(cv::KeyPoint(static_cast<float>(i * (640 / GRID_X)), static_cast<float>(j * (480 / GRID_Y)), 20));
                }
            }
        }
    }
    k.unlock();
    add_obstacles(obstacles);
}


void Planner::add_obstacles(std::vector<cv::KeyPoint> obstacles) {
    int obstacle_cnt = 0;
    for (auto& obstacle : obstacles) {
        // Add these obstacles to the obstacle_grid[x][y]
        // But first we need to convert coordiantes of obstacle[480][640] to coordinates of obstacle_grid[GRID_X][GRID_Y]
        // divide and round static_cast<int>(keypoint.pt.x)
        int scale_x = static_cast<int>(obstacle.pt.x / (640 / GRID_X));
        int scale_y = static_cast<int>(obstacle.pt.y / (480 / GRID_Y));

        // scale_x and scale_y are used in obstacle_grid
        planning_graph.obstacle_grid[scale_x][scale_y] = 1;
        obstacle_cnt += 1;
    }

    std::cout << obstacle_cnt << " obstacles added\n";
}


std::vector<std::pair<int, int>> Planner::backtrack(std::pair<int, int> start, std::pair<int, int> goal) {
    std::vector<std::pair<int, int>> path;

    std::pair<int, int> curr_node = start;

    const std::pair<int, int> directions[] = {
        {0, 1}, {0, -1}, {-1, 0}, {1, 0}
    };

    while (curr_node != goal) {
        path.push_back(curr_node);

        int min_value = std::numeric_limits<int>::max();
        std::pair<int, int> next_node;

        // Iterate through directions to find the next node with the minimum value
        for (const auto& dir : directions) {
            std::pair<int, int> neighbor = std::make_pair(curr_node.first + dir.first, curr_node.second + dir.second);

            // Check if neighbor is within grid bounds
            if (neighbor.first >= 0 && neighbor.first < GRID_X && neighbor.second >= 0 && neighbor.second < GRID_Y) {
                int neighbor_value = planning_graph.obstacle_grid[neighbor.first][neighbor.second];
                
                if (neighbor_value < min_value && neighbor_value > 1) {
                    min_value = neighbor_value;
                    next_node = neighbor;
                }
            }
        }

        curr_node = next_node;
    }

    // Reverse the path to have it in the correct order (from start to goal)
    //std::reverse(path.begin(), path.end());

    return path;

}



// Breadth first search through the 'graph'
// Assigns 
/*
while queue not empty
       node := first element of queue
       if node is what we are searching for
          return success
       endif
       //do whatever you need to do to node here
       children := find children of node in graph
       add children not in visited to back of fringe
       add node to visited
       remove node from fringe
    end while
    return failure
*/
void Planner::bfs(std::pair<int, int> start, std::pair<int, int> goal) {
    planning_graph.queue.push(goal);
    planning_graph.obstacle_grid[start.first][start.second] = 0;
    planning_graph.obstacle_grid[goal.first][goal.second] = 2;

    while (!planning_graph.queue.empty()) {
        std::pair<int, int> node = planning_graph.queue.front();
        if (node == start) {
            std::cout << "Start position found\n";
            break;
        }
        const std::pair<int, int> directions[] = {
                {0, 1}, {0, -1}, {-1, 0}, {1, 0}
        };

        for (const auto& dir : directions) {
            std::pair<int,int> child = std::make_pair(node.first + dir.first, node.second + dir.second);
            if (child.first >= 0 && child.first < GRID_X && child.second >= 0 && child.second < GRID_Y && planning_graph.obstacle_grid[child.first][child.second] != 1 && (std::find(planning_graph.visited.begin(), planning_graph.visited.end(), child) == planning_graph.visited.end())) {
                planning_graph.queue.push(child);
                planning_graph.visited.push_back(child);
                planning_graph.obstacle_grid[child.first][child.second] = planning_graph.obstacle_grid[node.first][node.second] + 1;
            }

        }
        planning_graph.visited.push_back(node);
        planning_graph.queue.pop();
    }
    
    
    for (int i = 0; i < GRID_Y; ++i) {
        for (int j = 0; j < GRID_X; ++j) {
            std::cout << std::setw(3) << planning_graph.obstacle_grid[j][i] << " ";
        }
        std::cout << '\n';
    }
    

    
}

