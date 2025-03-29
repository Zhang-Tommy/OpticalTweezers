#pragma once
#include "spot_manager.h"

void detect_beads();
int get_img(SpotManager* spotManager);
int get_img_offline_test(SpotManager* spotManager);
void bead_tracking(SpotManager* spotManager);