#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <queue>
#include "spot_manager.h"

#define GRID_X 53
#define GRID_Y 40
// Implementation of potential field motion planning algorithm in discrete grid space
// The individual cells (elements in grid) store a value (cost to goal wrt potential field)
// Computed via brushfire algorithm (Breadth first search)
// Wavefront planner steps:
// Mark goal state with 2, start state with 0
// Mark all obstacles with 1
// Brushfire away from the goal state (The energy at empty cells on frontier = current energy at frontier + change in cell energy)
// Then grid search to find navigation path to goal state
class Planner 
{
public:
	Planner(SpotManager* spotManager);
	Planner(SpotManager* spotManager, bool is_donut, bool is_line);
	SpotManager* spotManager;
	class Graph {
	public:
		int obstacle_grid[GRID_X][GRID_Y] = { 0 };
		std::queue<std::pair<int, int>> queue = {};
		std::vector<std::pair<int, int>> visited = {};
	};

	bool is_donut = false;
	bool is_line = false;

	Graph planning_graph = Graph();
	void bfs(std::pair<int, int> start, std::pair<int, int> goal);
	std::vector<std::pair<int, int>> backtrack(std::pair<int, int> start, std::pair<int, int> goal);
	void add_obstacles(std::vector<cv::KeyPoint> obstacles);
	void get_obstacles(std::pair<int, int> start, std::pair<int, int> goal);

};