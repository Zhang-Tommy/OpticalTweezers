#include "spot_manager.h"



// Spot Manager Constructor
// Sets all grid values to null and initializes hologram engine
SpotManager::SpotManager() {
	initialize_holo_engine();
}

// Returns pointer to an array holding all the spot parameters
// Spots are read from trapped_beads vector
std::vector<float> SpotManager::get_spots() {
	//float* spot_vals = new float[num_spots * 16];
	//std::unique_ptr<float[]> spot_vals = std::make_unique<float[]>(num_spots * 16);
	std::vector<float> spot_vals(num_spots * 16, 0.0f);
	int count = 0;

	for (const auto& entry : trapped_beads) {
		int x_pos = entry.first.first;
		int y_pos = entry.first.second;

		for (int j = 0; j < 16; j++) {
			spot_vals[count] = grid[x_pos][y_pos].vals[j];
			count += 1;
		}
	}

	return spot_vals;
}


// this compiles all the traps in the trapped_beads vector and sends over the spot data to hologram engine
int SpotManager::update_traps() {
	std::vector<float> raw_spot_vals = get_spots();
	int update_code = update_uniform(2, raw_spot_vals, sizeof(float) * num_spots * 4);

	return 0;
}

void SpotManager::create_trap(int x_pos, int y_pos) {
	//x(um), y, z, l (int), intenstity, phase
	// element 0 x  y  z  l    (x,y,z in um and l is an integer)
	// element 1 intensity (I) phase -  -
	// element 2 na.x na.y na.r -  (the x, y, and radius, of the spot on the SLM- useful for Shack-Hartmann holograms)
	// element 3 line trapping x y z and phase gradient.  xyz define the size and angle of the line, phase gradient (between +/-1) is the
	// scattering force component along the line.  Zero is usually a good choice for in-plane line traps
	float spot_params[16] = { x_pos * 0.1875, -y_pos * 0.1875, 0.0, 0.0,
							1.0, 0.0, 0.0, 0.0,
							0.0, 0.0, 1.0, 0.0,
							0.0, 0.0, 0.0, 0.0 };

	// create a new spot object
	Spot new_spot = Spot(spot_params);

	// add to the grid and trapped_beads data structures to keep track
	if (!grid[x_pos][y_pos].assigned) {
		grid[x_pos][y_pos] = new_spot;
		trapped_beads[std::make_pair(x_pos, y_pos)] = &grid[x_pos][y_pos];
		grid[x_pos][y_pos].assigned = true;
		num_spots += 1;
	}

	update_traps();
	// send over to the hologram engine
}

void SpotManager::create_trap(int x_pos, int y_pos, int z, int l, float i, float p) {
	//x(um), y, z, l (int), intenstity, phase
	// element 0 x  y  z  l    (x,y,z in um and l is an integer)
	// element 1 intensity (I) phase -  -
	// element 2 na.x na.y na.r -  (the x, y, and radius, of the spot on the SLM- useful for Shack-Hartmann holograms)
	// element 3 line trapping x y z and phase gradient.  xyz define the size and angle of the line, phase gradient (between +/-1) is the
	// scattering force component along the line.  Zero is usually a good choice for in-plane line traps
	float spot_params[16] = { x_pos * 0.1875, -y_pos * 0.1875, z, l,
							i, p, 0.0, 0.0,
							0.0, 0.0, 1.0, 0.0,
							0.0, 0.0, 0.0, 0.0 };
	
	// create a new spot object
	Spot new_spot = Spot(spot_params);

	// add to the grid and trapped_beads data structures to keep track
	if (!grid[x_pos][y_pos].assigned) {
		grid[x_pos][y_pos] = new_spot;
		trapped_beads[std::make_pair(x_pos, y_pos)] = &grid[x_pos][y_pos];
		grid[x_pos][y_pos].assigned = true;
		num_spots += 1;
	}

	update_traps();
	// send over to the hologram engine
}

// Moves the trap from point a to point b
void SpotManager::move_trap(int x_trap, int y_trap, int x_new, int y_new) {
	// check for existing trap at x_new, y_new

	if (grid[x_new][y_new].assigned) {
		std::cout << "Trap already exists at " << x_new << ", " << y_new << std::endl;
	}
	else if (!grid[x_trap][y_trap].assigned) {
		std::cout << "Trap does not exist at " << x_trap << ", " << y_trap << std::endl;
	}
	else {
		// copy over the spot to new grid position
		grid[x_new][y_new] = grid[x_trap][y_trap];
		
		// unassign and clear the trap from previous grid position
		grid[x_trap][y_trap].assigned = false;
		grid[x_trap][y_trap].clear();
		trapped_beads.erase(std::make_pair(x_trap, y_trap));

		// change the x and y position in raw spot values
		grid[x_new][y_new].set_new_pos(x_new, y_new);
		trapped_beads[std::make_pair(x_new, y_new)] = &grid[x_new][y_new];
		update_traps();
	}
}

void SpotManager::remove_trap(int x_pos, int y_pos) {
	// check if trap exists

	if (grid[x_pos][y_pos].assigned) {
		grid[x_pos][y_pos].assigned = false;
		grid[x_pos][y_pos].clear();
		num_spots -= 1;
		trapped_beads.erase(std::make_pair(x_pos, y_pos));
	}
	else {
		std::cout << "Attempted to remove a trap that does not exist" << std::endl;
	}

}