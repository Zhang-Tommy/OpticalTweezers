
#include "motion_planner.h"
#include <iostream>
#include <iomanip>

#define GRID_X 25
#define GRID_Y 25


Planner::Planner() {
    srand(static_cast<unsigned>(time(nullptr))); // Seed for random number generation
    int num_obstacles = GRID_X * 4;

    for (int i = 0; i < num_obstacles; ++i) {
        int rand_x = rand() % GRID_X; // Random x-coordinate between 0 and 50
        int rand_y = rand() % GRID_Y; // Random y-coordinate between 0 and 50

        planning_graph.obstacle_grid[rand_x][rand_y] = 1;
    }
    
	
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
                
                if (neighbor_value < min_value && neighbor_value != 1) {
                    min_value = neighbor_value;
                    next_node = neighbor;
                }
            }
        }
        
        curr_node = next_node;
    }

    // Reverse the path to have it in the correct order (from start to goal)
    std::reverse(path.begin(), path.end());

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
    
    for (int i = 0; i < GRID_X; ++i) {
        for (int j = 0; j < GRID_Y; ++j) {
            std::cout << std::setw(3) << planning_graph.obstacle_grid[i][j] << " ";
        }
        std::cout << '\n';
    }
    

    
}

