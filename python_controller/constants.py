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
ANGLE = -3 * (math.pi/180)
Z_OFFSET = -8.0
SCALE_X = 1.10
SCALE_Y = 1.10

""" Path to GIGE camera configuration file"""
CTI_FILE_DIR = r'C:\Users\User\Desktop\Tommy_Tweezers_Automation\tweezers_automation\tweezers_automation_v2\bgapi2_gige.cti'

""" Maximum number of traps in workspace"""
MAX_NUM_TRAPS = 5

""" RK4 Integration time step for bead dynamics"""
DT = 1e-3

""" Planning horizon steps"""
N = 85
# 50

""" Bead radius in pixels"""
BEAD_RADIUS = 13

""" Default line trap length"""
LINE_TRAP_LENGTH = 3
LINE_TRAP_ANGLE = math.pi / 2

""" Annular trap vortex charge/twist """
ANNULAR_TRAP_VORTEX_CHARGE = 9

""" Obstacle separation boundary"""
OBSTACLE_SEPARATION = 26 + 7

OBSTACLE_SEPARATION_OG = 26 + 7

DONUT_Z_OFFSET = 0

""" Constant keypoints array size (so jax doesn't recompile and make it slow)"""
KPS_SIZE = 150

"""ILQR Max Iteration Count"""
MAX_ILQR_ITERATIONS = 150