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


Spot Spot::set_spot(float x, float y, float z, int l, float i, float p) {
    // x(um), y, z, l (int), intenstity, phase
	// element 0 x  y  z  l    (x,y,z in um and l is an integer)
	// element 1 intensity (I) phase -  -
	// element 2 na.x na.y na.r -  (the x, y, and radius, of the spot on the SLM- useful for Shack-Hartmann holograms)
	// element 3 line trapping x y z and phase gradient.  xyz define the size and angle of the line, phase gradient (between +/-1) is the
	// scattering force component along the line.  Zero is usually a good choice for in-plane line traps
	float spot_params[16] = { y, -x, z, l, 
							i, p, 0.0, 0.0, 
							0.0, 0.0, 0.0, 0.0,
							0.0, 0.0, 0.0, 0.0 };
	set_spot_values(spot_params);
	return NULL;

	return NULL;
}

Spot Spot::set_line_trap(float x_pos, float y_pos, float x_len, float y_len, float phase_grad) {
	//line trapping x,y,z, phase gradient
	float spot_params[16] = { y_pos, -x_pos, 0.0, 0.0,
							1.0, 0.0, 0.0, 0.0,
							0.0, 0.0, 0.0, 0.0,
							x_len, y_len, 0.0, phase_grad };
	set_spot_values(spot_params);
	return NULL;
}

Spot Spot::set_na(float x_pos, float y_pos, float na_x, float na_y, float na_r) {
	//na.x, na.y, na.z
	float spot_params[16] = { y_pos, -x_pos, 0.0, 0.0,
							1.0, 0.0, 0.0, 0.0,
							na_x, na_y, na_r, 0.0,
							0.0, 0.0, 0.0, 0.0 };
	set_spot_values(spot_params);
	return NULL;
}




