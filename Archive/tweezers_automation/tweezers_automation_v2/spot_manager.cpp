#include "spot_manager.h"
#include <ctime>
#include <thread>
#include <cmath>
#include <numbers>
#include <mutex>

constexpr float SCALE_X = 0.1875 * 1.07;
constexpr float SCALE_Y = 0.1875 * 1.10;
constexpr float ANGLE = 3 * (std::numbers::pi / 180);
constexpr float Z_OFFSET = -8.0;

extern std::mutex t;

// Spot Manager Constructor
// Sets all grid values to null and initializes hologram engine
SpotManager::SpotManager() {
	initialize_holo_engine();
}

// Returns pointer to an array holding all the spot parameters
// Spots are read from trapped_beads vector
std::vector<float> SpotManager::get_spots() {
	std::vector<float> spot_vals(num_spots * 16, 0.0f);
	int count = 0;
	t.lock();
	for (const auto& entry : trapped_beads) {
		int x_pos = entry.first.first;
		int y_pos = entry.first.second;

		for (int j = 0; j < 16; j++) {
			spot_vals[count] = grid[x_pos][y_pos].vals[j];
			count += 1;
		}
	}
	t.unlock();
	return spot_vals;
}

// Updates hologram engine with current spot parameters
int SpotManager::update_traps() {
	std::vector<float> raw_spot_vals = get_spots();
	int update_code = update_uniform(2, raw_spot_vals, sizeof(float) * num_spots * 4);

	if (update_code != 0) {
		return 0;
	}
	else {
		std::cout << "Failed to update uniform variables!\n";
	}
}

void SpotManager::create_trap(int x_pos, int y_pos) {
	// x(um), y, z, l (int), intenstity, phase
	// element 0 x  y  z  l    (x,y,z in um and l is an integer)
	// element 1 intensity (I) phase -  -
	// element 2 na.x na.y na.r -  (the x, y, and radius, of the spot on the SLM- useful for Shack-Hartmann holograms)
	// element 3 line trapping x y z and phase gradient.  xyz define the size and angle of the line, phase gradient (between +/-1) is the
	// scattering force component along the line.  Zero is usually a good choice for in-plane line traps
	float x_new = (x_pos * SCALE_X) * cos(ANGLE) - (y_pos * SCALE_Y) * sin(ANGLE);
	float y_new = -((x_pos * SCALE_X) * sin(ANGLE) + (y_pos * SCALE_Y) * cos(ANGLE));
	float z_new = Z_OFFSET;

	float spot_params[16] = { x_new, y_new, z_new, 0.0,
							1.0, 0.0, 0.0, 0.0,
							0.0, 0.0, 1.0, 0.0,
							0.0, 0.0, 0.0, 0.0 };

	// create a new spot object
	Spot new_spot = Spot(spot_params);

	// add to the grid and trapped_beads data structures to keep track
	if (!grid[x_pos][y_pos].assigned) {
		grid[x_pos][y_pos] = new_spot;
		t.lock();
		trapped_beads[std::make_pair(x_pos, y_pos)] = &grid[x_pos][y_pos];
		t.unlock();
		grid[x_pos][y_pos].assigned = true;
		num_spots += 1;
	}

	update_traps();
}

void SpotManager::create_trap(int x_pos, int y_pos, int z, int l, float i, float p) {
	float x_new = (x_pos * SCALE_X) * cos(ANGLE) - (y_pos * SCALE_Y) * sin(ANGLE);
	float y_new = -((x_pos * SCALE_X) * sin(ANGLE) + (y_pos * SCALE_Y) * cos(ANGLE));
	float z_new = Z_OFFSET;

	float spot_params[16] = { x_new, y_new, z + z_new, l,
							i, p, 0.0, 0.0,
							0.0, 0.0, 1.0, 0.0,
							0.0, 0.0, 0.0, 0.0 };

	// create a new spot object
	Spot new_spot = Spot(spot_params);

	// add to the grid and trapped_beads data structures to keep track
	if (!grid[x_pos][y_pos].assigned) {
		grid[x_pos][y_pos] = new_spot;
		t.lock();
		trapped_beads[std::make_pair(x_pos, y_pos)] = &grid[x_pos][y_pos];
		t.unlock();
		grid[x_pos][y_pos].assigned = true;
		num_spots += 1;
	}

	update_traps();
	// send over to the hologram engine
}

// Create an annular trap at specified position and na parameters
void SpotManager::create_donut(int x_pos, int y_pos, int vortex_charge, int z) {
	float x_new = (x_pos * SCALE_X) * cos(ANGLE) - (y_pos * SCALE_Y) * sin(ANGLE);
	float y_new = -((x_pos * SCALE_X) * sin(ANGLE) + (y_pos * SCALE_Y) * cos(ANGLE));
	float z_new = Z_OFFSET + z;

	float spot_params[16] = { x_new, y_new, z_new, vortex_charge,
							1.0, 0.0, 0.0, 0.0,
							0.0, 0.0, 0.0, 0.0,
							0.0, 0.0, 0.0, 0.0 };

	// create a new spot object
	Spot new_spot = Spot(spot_params);

	// add to the grid and trapped_beads data structures to keep track
	if (!grid[x_pos][y_pos].assigned) {
		grid[x_pos][y_pos] = new_spot;
		t.lock();
		trapped_beads[std::make_pair(x_pos, y_pos)] = &grid[x_pos][y_pos];
		t.unlock();
		grid[x_pos][y_pos].assigned = true;
		num_spots += 1;
	}

	update_traps();
	// send over to the hologram engine
}

// Create a line trap at specified position with desired length in x/y directions
void SpotManager::create_line(int x_pos, int y_pos, int x_len, int y_len) {
	float x_new = (x_pos * SCALE_X) * cos(ANGLE) - (y_pos * SCALE_Y) * sin(ANGLE);
	float y_new = -((x_pos * SCALE_X) * sin(ANGLE) + (y_pos * SCALE_Y) * cos(ANGLE));
	float z_new = Z_OFFSET;

	float spot_params[16] = { x_new, y_new, z_new, 0.0,
							1.0, 0.0, 0.0, 0.0,
							0.0, 0.0, 1.0, 0.0,
							x_len, y_len, 0.0, 0.0 };

	// create a new spot object
	Spot new_spot = Spot(spot_params);

	// add to the grid and trapped_beads data structures to keep track
	if (!grid[x_pos][y_pos].assigned) {
		grid[x_pos][y_pos] = new_spot;
		t.lock();
		trapped_beads[std::make_pair(x_pos, y_pos)] = &grid[x_pos][y_pos];
		t.unlock();
		grid[x_pos][y_pos].assigned = true;
		num_spots += 1;
	}

	update_traps();
	// send over to the hologram engine
}

const std::vector<std::pair<int, int>> SpotManager::interpolate_path(float x_trap, float y_trap, float x_new, float y_new) {
	float dist = hypotf(x_trap - x_new, y_trap - y_new) * 0.1875;
	int num_segments = dist * 2;
	float dx = (x_new - x_trap) / num_segments;
	float dy = (y_new - y_trap) / num_segments;
	float m;
	if (x_new - x_trap == 0) {
		m = (y_new - y_trap);
	}
	else {
		m = (y_new - y_trap) / (x_new - x_trap);
	}

	std::vector<std::pair<int, int>> path_sequence;

	for (int i = 0; i < num_segments; i++) {
		int int_x = static_cast<int>(x_trap + i * dx);
		int int_y;
		if (x_new - x_trap == 0) {
			int_y = static_cast<int>(y_trap + i * dy);
		}
		else {
			int_y = static_cast<int>(y_trap + i * dx * m);
		}
		path_sequence.push_back(std::make_pair(int_x, int_y));
	}

	path_sequence.pop_back();
	path_sequence.emplace_back(std::make_pair(x_new, y_new));
	path_sequence.erase(path_sequence.begin());

	return path_sequence;
}



// Move the specified trap at x_trap, y_trap to x_new, y_new in a straight, shortest path line
void SpotManager::translate_trap(float x_trap, float y_trap, float x_new, float y_new, float um_sec) {
	std::vector<std::pair<int, int>> path_sequence = interpolate_path(x_trap, y_trap, x_new, y_new);

	float delay = 0.5 / um_sec;

	std::pair prev_point = std::make_pair(static_cast<int>(x_trap), static_cast<int>(y_trap));

	for (std::pair point : path_sequence) {
		move_trap(prev_point.first, prev_point.second, point.first, point.second, true);
		prev_point = point;
		std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<long>(delay * 1000)));
	}
}

// Moves the trap from point a to point b in one step
void SpotManager::move_trap(int x_trap, int y_trap, int x_new, int y_new, bool update) {
	if (x_new > 480 || y_new > 640 || x_new < 0 || y_new < 0) {
		throw;
	}
	if (grid[x_new][y_new].assigned) {
		if (x_trap == x_new || y_trap == y_new) {
			std::cout << "Trap cannot move to the same coordinate \n";
		}
		else {
			std::cout << "Trap already exists at " << x_new << ", " << y_new << std::endl;
		}
		
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
		t.lock();
		trapped_beads.erase(std::make_pair(x_trap, y_trap));

		// change the x and y position in raw spot values
		grid[x_new][y_new].set_new_pos(x_new, y_new);
		//grid[x_new][x_new].assigned = true;
		trapped_beads[std::make_pair(x_new, y_new)] = &grid[x_new][y_new];
		t.unlock();
		if (update) {
			update_traps();
		}
	}
}

void SpotManager::remove_trap(int x_pos, int y_pos) {
	if (grid[x_pos][y_pos].assigned) {
		grid[x_pos][y_pos].assigned = false;
		grid[x_pos][y_pos].clear();
		num_spots -= 1;
		t.lock();
		trapped_beads.erase(std::make_pair(x_pos, y_pos));
		t.unlock();
	}
	else {
		std::cout << "Attempted to remove a trap that does not exist" << std::endl;
	}

}