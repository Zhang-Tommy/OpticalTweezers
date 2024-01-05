#pragma once


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
};