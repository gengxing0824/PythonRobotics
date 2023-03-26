import numpy as np
from math import tan

# for car 
WB = 3.0  # rear to front wheel
W = 2.0  # width of car
LF = 3.3  # distance from rear to vehicle front end
LB = 1.0  # distance from rear to vehicle back end
MAX_STEER = 0.6  # [rad] maximum steering angle
MAX_CURVATURE = tan(MAX_STEER) / WB  # 自行车模型


# for hybird_a_star
XY_GRID_RESOLUTION = 0.1  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(1.0)  # [rad]
MOTION_RESOLUTION = 0.1  # [m] path interpolate resolution
N_STEER = 3  # number of steer command
OBST_GRID_RESOLUTION = 0.05

SB_COST = 1000.0  # switch back penalty cost 换档的代价
BACK_COST = 1.0  # backward penalty cost
STEER_CHANGE_COST = 20.0  # steer angle change penalty cost
STEER_COST = 0.0  # steer angle change penalty cost
H_COST = 10.0  # Heuristic costq

