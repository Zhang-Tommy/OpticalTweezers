#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "udp_sender.h"
#include "spot.h"

// Spot manager holds functionalities for creating, modifying, and moving traps
class SpotManager
{
public:
    SpotManager();
    Spot grid[480][640];
    void create_trap(int x_pos, int y_pos);
    void create_trap(int x, int y, int z, int l, float i, float p);
    void move_trap(int x_trap, int y_trap, int x_new, int y_new);
    void remove_trap(int x_pos, int y_pos);
    int update_traps();
    
    std::map<std::pair<int, int>, Spot*> trapped_beads;
    int num_spots = 0;
    
private:
    std::vector<float> get_spots();
};

