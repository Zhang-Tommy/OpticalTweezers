from spot_manager import SpotManager
from utilities import *

# Initialize the hologram engine (sends commands to spatial light modulator that generates holograms)
init_holo_engine()

# Initialize spot manager (create, move, modify optical traps)
sm = SpotManager()

# Test operation of spot manager
x = 300
y = 300

sm.add_spot((x,y))

for i in range(100):
    time.sleep(0.1)
    sm.move_trap((x, y), (x+1, y+1))
    x = x+1
    y = y+1


