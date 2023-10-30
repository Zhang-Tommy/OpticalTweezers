#include "spot.h"


Spot::Spot(float new_vals[]) {
	set_spot_values(new_vals);
}

Spot::Spot() {
	float zeros[16] = { 0.0 };
	set_spot_values(zeros);
}

// this needs to also send over the values to the hologram engine
void Spot::set_spot_values(float new_vals[]) {
	for (int i = 0; i < 16; i++) {
		vals[i] = new_vals[i];
	}
}
