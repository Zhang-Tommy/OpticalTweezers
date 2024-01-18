
#include "motion_planner.h"
#include <iostream>
// represent work area as grid
// define class for objects in the grid
// what do we need for motion planning?
// we need to know locations of obstacles
// we need to know location of start and end points
// 

/*
  initialize:
    graph := {nodes}, {edges}
    fringe := {root}
    visited := empty

  breadth-first-search (graph, fringe, visited):
    while fringe not empty
       node := first element of fringe
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


Planner::Planner() {
    

    // I would actually set the obstacles, goal, start states here
    // Manually setting the goal, start, and obstacles:
    planning_graph.obstacle_grid[0][0] = 0;
    planning_graph.obstacle_grid[0][300] = 2;
    planning_graph.obstacle_grid[0][299] = 1;
    
	
}

void Planner::get_obstacles() {
    // set of keypoints - trapped_beads
}

void Planner::set_goal() {
    // Wherever I want to move the bead to
}

// Breadth first search through the 'graph'
// Assigns 
/*
while fringe not empty
       node := first element of fringe
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
void Planner::bfs(std::pair<int, int> goal) {
    while (!planning_graph.queue.empty()) {
        std::pair<int, int> node = planning_graph.queue.front();
        
        if (node == goal) {
            std::cout << "bfs found goal state";
            break;
        }
        // do something to the node

        // get the children of the current node
        // add the node to visited
        // remove node from fringe

    }
}
