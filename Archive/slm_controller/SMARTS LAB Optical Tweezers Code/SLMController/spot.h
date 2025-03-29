#pragma once

#include <iostream>
#include <string>

class Spot
{
public:
    void set_spot_values(float new_vals[]);
    Spot(float new_vals[]);
    Spot();
    float vals[16];
};

