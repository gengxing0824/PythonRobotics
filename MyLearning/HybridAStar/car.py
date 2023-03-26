"""

Car model for Hybrid A* path planning

author: Zheng Zh (@Zhengzh)

"""

import sys
import pathlib
import math
from turtle import color

from sqlalchemy import false, true
root_dir = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))
from config import *
from math import cos, sin, tan, pi

import matplotlib.pyplot as plt
import numpy as np

from utils.angle import rot_mat_2d

WB = 3.0  # rear to front wheel
W = 2.0  # width of car
LF = 3.3  # distance from rear to vehicle front end
LB = 1.0  # distance from rear to vehicle back end
MAX_STEER = 0.6  # [rad] maximum steering angle
MAX_CURVATURE = tan(MAX_STEER) / WB  # 自行车模型

BUBBLE_DIST = (LF - LB) / 2.0  # distance from rear to center of vehicle.
BUBBLE_R = np.hypot((LF + LB) / 2.0, W / 2.0)  # bubble radius

# vehicle rectangle vertices
VRX = [LF, LF, -LB, -LB, LF]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]

VRX = [3.183, 3.423, 3.592, 3.660, 3.592, 3.423, 3.183, -0.574, -0.853, -0.922, -0.960, -0.922, -0.853, -0.547, 3.183]
VRY = [0.880, 0.658, 0.313, 0, -0.313, -0.658, -0.880, -0.850, -0.654, -0.235, 0, 0.235, 0.654, 0.850, 0.880]


def check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
    for i_x, i_y, i_yaw in zip(x_list, y_list, yaw_list):
        cx = i_x + BUBBLE_DIST * cos(i_yaw)
        cy = i_y + BUBBLE_DIST * sin(i_yaw)

        ids = kd_tree.query_ball_point([cx, cy], BUBBLE_R)

        if not ids:
            continue

        if not rectangle_check(i_x, i_y, i_yaw,
                               [ox[i] for i in ids], [oy[i] for i in ids]):
            return False  # collision

    return True  # no collision

def check_car_collision_dist_map(x_list, y_list, yaw_list, dist_map, config):
    for i_x, i_y, i_yaw in zip(x_list, y_list, yaw_list):
        c, s = cos(i_yaw), sin(i_yaw)
        rot = rot_mat_2d(-i_yaw)
        for rx, ry in zip(VRX, VRY):
            converted_xy = np.stack([rx, ry]).T @ rot
            x = converted_xy[0]+i_x
            y = converted_xy[1]+i_y
            if (x >= config.max_x or
                y >= config.max_y or
                x <= config.min_x or
                y <= config.min_y or
                dist_map[math.floor(y/config.obst_grid_resolution), math.floor(x/config.obst_grid_resolution)] < 0.05):
                return False
    return True  # no collision


def rectangle_check(x, y, yaw, ox, oy):
    # transform obstacles to base link frame
    rot = rot_mat_2d(yaw)
    for iox, ioy in zip(ox, oy):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        rx, ry = converted_xy[0], converted_xy[1]

        if not (rx > LF or rx < -LB or ry > W / 2.0 or ry < -W / 2.0):
            return False  # no collision

    return True  # collision

def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """Plot arrow."""
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(x, y, length * cos(yaw), length * sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)

def plot_car(x, y, yaw, cc = 1):
    car_color = '-k'
    c, s = cos(yaw), sin(yaw)
    rot = rot_mat_2d(-yaw)
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(VRX, VRY):
        rx *= cc
        ry *= cc
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0]+x)
        car_outline_y.append(converted_xy[1]+y)

    arrow_x, arrow_y, arrow_yaw = x, y, yaw
    plot_arrow(arrow_x, arrow_y, arrow_yaw, length = cc, width= cc*0.5)

    plt.plot(car_outline_x, car_outline_y, car_color)
    


def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi


def move(x, y, yaw, distance, steer, L=WB):
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    yaw += pi_2_pi(distance * tan(steer) / L)  # distance/2

    return x, y, yaw


def main():
    x, y, yaw = 0., 0., 1.
    plt.axis('equal')
    plot_car(x, y, yaw, 3)
    plt.show()


if __name__ == '__main__':
    main()
