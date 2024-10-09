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
SCALE_X = 1.07
SCALE_Y = 1.10

""" Path to GIGE camera configuration file"""
CTI_FILE_DIR = r'C:\Users\User\Desktop\Tommy_Tweezers_Automation\tweezers_automation\tweezers_automation_v2\bgapi2_gige.cti'

""" Maximum number of traps in workspace"""
MAX_NUM_TRAPS = 5

""" RK4 Integration time step for bead dynamics"""
DT = 1e-3

""" Planning horizon steps"""
N = 125
# 50

""" Bead radius in pixels"""
BEAD_RADIUS = 20

""" Default line trap length"""
LINE_TRAP_LENGTH = 8
LINE_TRAP_ANGLE = 0

""" Annular trap vortex charge/twist """
ANNULAR_TRAP_VORTEX_CHARGE = 12