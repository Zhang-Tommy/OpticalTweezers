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
    std::map<std::pair<int, int>, Spot*> trapped_beads;
    int num_spots = 0;

    void create_trap(int x_pos, int y_pos);
    void create_trap(int x_pos, int y_pos, int z, int l, float i, float p);
    void create_donut(int x_pos, int y_pos, int vortex_charge, float na_r);
    void create_line(int x_pos, int y_pos, int x_len, int y_len);
    void move_trap(int x_trap, int y_trap, int x_new, int y_new);
    void translate_trap(float x_trap, float y_trap, float x_new, float y_new, float um_sec);
    void remove_trap(int x_pos, int y_pos);
    int update_traps();

private:
    std::vector<float> get_spots();
};

