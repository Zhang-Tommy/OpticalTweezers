#include "spot.h"
#include <numbers>

constexpr float SCALE_X = 0.1875 * 1.07;
constexpr float SCALE_Y = 0.1875 * 1.10;
constexpr float ANGLE = 3 * (std::numbers::pi / 180);
constexpr float Z_OFFSET = -8.0;

// Spot Constructor with default zeros
Spot::Spot() {
	float zeros[16] = { 0.0 };
	set_spot_values(zeros);
	
}

// Spot constructor with specified values
Spot::Spot(float new_vals[]) {
	set_spot_values(new_vals);
}

// Modifies spot value array
void Spot::set_spot_values(float new_vals[]) {
	for (int i = 0; i < 16; i++) {
		vals[i] = new_vals[i];
	}
}

// Modifies spot value array
void Spot::set_new_pos(int x_pos, int y_pos) {
	float x_new = (x_pos * SCALE_X) * cos(ANGLE) - (y_pos * SCALE_Y) * sin(ANGLE);
	float y_new = -((x_pos * SCALE_X) * sin(ANGLE) + (y_pos * SCALE_Y) * cos(ANGLE));

	vals[0] = x_new;
	vals[1] = y_new;
}

// Sets all values of a spot to zero
void Spot::clear() {
	float zeros[16] = { 0.0 };
	for (int i = 0; i < 16; i++) {
		vals[i] = zeros[i];
	}
}

