#pragma once

#include <iostream>
#include <string>

class Spot
{
public:
    void set_spot_values(float new_vals[]);
    Spot(float new_vals[]);
    Spot();
    Spot set_spot(float x, float y, float z, int l, float i, float p);
    Spot set_line_trap(float x_pos, float y_pos, float x_len, float y_len, float phase_grad);
    Spot set_na(float x_pos, float y_pos, float na_x, float na_y, float na_r);
    float vals[16];
};

