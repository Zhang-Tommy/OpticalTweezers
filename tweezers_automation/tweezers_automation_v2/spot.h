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
    PID *pid_x = new PID(0.3, 480, -480, 0.1, 0.01, 0.5);
    PID *pid_y = new PID(0.3, 640, -640, 0.1, 0.01, 0.5);
};