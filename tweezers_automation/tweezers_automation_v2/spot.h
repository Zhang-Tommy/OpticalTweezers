#pragma once

#include "pid.h"

// Spot represents parameters needed to send over to holo engine
// in order to generate a trap
class Spot
{
public:
    Spot(float new_vals[]);
    Spot();
    void set_spot_values(float new_vals[]);
    void set_new_pos(int x_pos, int y_pos);
    void clear();
    float vals[16];
    bool assigned = false;
    PID *pid_x = new PID(0.3, 30, -30, 0.1, 0.01, 0.5);
    PID *pid_y = new PID(0.3, 30, -30, 0.1, 0.01, 0.5);
    bool is_donut = false;
    bool is_line = false;
};

// How to add donut and line traps?
// Detecting beads, depending if we want automatic or manual donut trap creation
// Bead tracking will also need to use the detected beads
// if there are beads in the radius / near the radius of the donut trap we consider those to be with the donut trap
// from those beads, calculate the centroid and you can use that as the bead position for PID control. Similarly for line traps
// For motion planning, this needs to account for the size ofthe donut trap. Or implement the algorithm which does not sit nearest to the obstacles.