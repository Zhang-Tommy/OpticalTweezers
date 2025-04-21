import math

""" PORT and IP address for hologram engine UDP packets"""
UDP_PORT = 61557
UDP_IP = '127.0.0.1'

""" Camera dimensions"""
CAM_X = 640
CAM_Y = 480

""" Scaling factor to convert from pixels to micrometers"""
CAM_TO_UM = 0.1875

""" Calibration factors for optical misalignment"""
# ANGLE = -3 * (math.pi/180)
# Z_OFFSET = 0.0 # -8.0 std val
# SCALE_X = 1.10
# SCALE_Y = 1.10
ANGLE = 0
Z_OFFSET = 0.0 # -8.0 std val
SCALE_X = 1
SCALE_Y = 1

""" Path to GIGE camera configuration file"""
CTI_FILE_DIR = r'C:\Users\User\Desktop\Tommy_Tweezers_Automation\tweezers_automation\tweezers_automation_v2\bgapi2_gige.cti'
"""Parameters when removing trapped beads from obstacles"""
R_DONUT = 35
R_POINT = 18
L_WIDTH = 35
L_LENGTH = 75

""" Minimum distance to goal in which controller will disregard obstacles"""
GOAL_DIST_OBSTACLE_FREE = 100 # pixels

""" Maximum number of traps in workspace"""
MAX_NUM_TRAPS = 5

""" RK4 Integration time step for bead dynamics"""
DT = 1e-3  #1e-3

""" Planning horizon steps"""
N = 85  # 85
MPC_COMPUTE_FREQ = 2 # Number of time steps until recompute state/control sequence (minimum val 1) -> high vals = less compute required

""" Debug mode"""
DEBUG = True
SIMULATOR_MODE = True

""" Bead radius in pixels"""
BEAD_RADIUS = 13

""" Default line trap length"""
LINE_TRAP_LENGTH = 6
LINE_TRAP_ANGLE = math.pi / 2

""" Line trap elliptical boundary """
ELLIPSE_MINOR_AXIS = 12
ELLIPSE_MAJOR_AXIS = 30 # pixels

""" Annular trap vortex charge/twist """
ANNULAR_TRAP_VORTEX_CHARGE = 2

""" Obstacle separation boundary"""
OBSTACLE_SEPARATION = 26 + 7

OBSTACLE_SEPARATION_OG = 26 + 7

#DONUT_Z_OFFSET = -4.0
DONUT_Z_OFFSET = 0
""" Constant keypoints array size (so jax doesn't recompile and make it slow)"""
KPS_SIZE = 50

"""ILQR Max Iteration Count"""
MAX_ILQR_ITERATIONS = 15

""" Desired Average Laser Power (mW)"""
AVG_DESIRED_LASER_PWR = 250  # Min laser output is 950mW
TOTAL_LASER_PWR = 1000

