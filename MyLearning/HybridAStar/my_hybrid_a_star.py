import heapq
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import sys
import pathlib
from config import *
from get_map import get_map
from dist_map import get_dist_map
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from dynamic_programming_heuristic import calc_distance_heuristic
from ReedsSheppPath import reeds_shepp_path_planning as rs
# import reeds_shepp_path_planning as rs
from car import move, check_car_collision_dist_map, MAX_STEER, WB, plot_car, BUBBLE_R, MAX_CURVATURE
MAX_FIND_PATH_NUM = 10
show_animation = 1
show_animation_end = 1

start= [10.5, 2, np.deg2rad(85.0)]
goal = [10, 10, np.deg2rad(10.0)]
image_path = "MyLearning\\HybridAStar\\fusion_map_2023-03-19-04-11-11.jpg"

class Path:
    def __init__(self, x_list, y_list, yaw_list, direction_list, cost):
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.direction_list = direction_list
        self.cost = cost

class Node:
    def __init__(self, x_ind, y_ind, yaw_ind, direction,
                 x_list, y_list, yaw_list, directions,
                 steer=0.0, parent_index=None, cost=None):
        self.x_index = x_ind
        self.y_index = y_ind
        self.yaw_index = yaw_ind
        self.direction = direction  # true: D ; false: R
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.directions = directions
        self.steer = steer
        self.parent_index = parent_index
        self.cost = cost


class Config:
    def __init__(self, dist_map, xy_resolution, yaw_resolution, obst_grid_resolution):
        min_x_m = 0
        min_y_m = 0
        max_x_m = dist_map.shape[1] * obst_grid_resolution
        max_y_m = dist_map.shape[0] * obst_grid_resolution

        self.min_x = min_x_m
        self.min_y = min_y_m
        self.max_x = max_x_m
        self.max_y = max_y_m
        self.min_x_index = round(min_x_m/xy_resolution)
        self.min_y_index = round(min_y_m/xy_resolution)
        self.max_x_index = round(max_x_m/xy_resolution)
        self.max_y_index = round(max_y_m/xy_resolution)
        self.x_w = round(self.max_x - self.min_x)
        self.y_w = round(self.max_y - self.min_y)

        self.min_yaw = round(- math.pi / yaw_resolution) - 1
        self.max_yaw = round(math.pi / yaw_resolution)
        self.yaw_w = round(self.max_yaw - self.min_yaw)

        self.xy_resolution = xy_resolution
        self.yaw_resolution = yaw_resolution
        self.obst_grid_resolution = obst_grid_resolution

def mod2pi(x):
    # Be consistent with fmod in cplusplus here.
    v = np.mod(x, np.copysign(2*math.pi, x))
    if v < 0:
        v = 2.0 * math.pi + v
    return v

def calc_index(node, c):
    node.yaw_index = mod2pi(node.yaw_index)
    ind = node.x_index//c.xy_resolution *1000000 + node.y_index//c.xy_resolution*10000 + node.yaw_index//c.yaw_resolution
    return ind

def calc_rs_path_cost(reed_shepp_path):
    # print(reed_shepp_path.lengths)
    cost = 0.0
    for length in reed_shepp_path.lengths:
        if length >= 0:  # forward
            cost += length
        else:  # back
            cost += abs(length) * BACK_COST

    # switch back penalty
    for i in np.arange(len(reed_shepp_path.lengths) - 1):
        # switch back
        if reed_shepp_path.lengths[i] * reed_shepp_path.lengths[i + 1] < 0.0:
            cost = cost + SB_COST

    # steer penalty
    for course_type in reed_shepp_path.ctypes:
        if course_type != "S":  # curve
            cost += STEER_COST * abs(MAX_STEER)

    # ==steer change penalty
    # calc steer profile
    n_ctypes = len(reed_shepp_path.ctypes)
    u_list = [0.0] * n_ctypes
    for i in np.arange(n_ctypes):
        if reed_shepp_path.ctypes[i] == "R":
            u_list[i] = - MAX_STEER
        elif reed_shepp_path.ctypes[i] == "L":
            u_list[i] = MAX_STEER

    for i in np.arange(len(reed_shepp_path.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(u_list[i + 1] - u_list[i])

    return cost
    
def analytic_expansion(current, goal, dist_map, config):
    start_x = current.x_list[-1]
    start_y = current.y_list[-1]
    start_yaw = current.yaw_list[-1]

    goal_x = goal.x_list[-1]
    goal_y = goal.y_list[-1]
    goal_yaw = goal.yaw_list[-1]

    # max_curvature = math.tan(MAX_STEER) / WB
    max_curvature = MAX_CURVATURE
    paths = rs.calc_paths(start_x, start_y, start_yaw,
                          goal_x, goal_y, goal_yaw,
                          max_curvature, step_size=MOTION_RESOLUTION)

    if not paths:
        return None

    best_path, best = None, None

    for path in paths:
        if check_car_collision_dist_map(path.x, path.y, path.yaw, dist_map, config):   # 碰撞检测
            if (path.lengths[0]*current.direction < 0.0):
                cost = calc_rs_path_cost(path) + SB_COST
            else:
                cost = calc_rs_path_cost(path)
            if not best or best > cost:
                best = cost
                best_path = path

    return best_path
    
def update_node_with_analytic_expansion(current, goal, config, dist_map):
    path = analytic_expansion(current, goal, dist_map, config)

    if path:
        # if show_animation:
            # plt.plot(np.array(path.x)/config.obst_grid_resolution, np.array(path.y)/config.obst_grid_resolution)
            # plt.pause(0.001)
        f_x = path.x[1:]
        f_y = path.y[1:]
        f_yaw = path.yaw[1:]
        f_d = path.directions[1:]

        if (path.lengths[0]*current.direction < 0.0):
            f_cost = current.cost + calc_rs_path_cost(path) + SB_COST
        else:
            f_cost = current.cost + calc_rs_path_cost(path) 
        f_parent_index = calc_index(current, config)

        f_steer = 0.0
        f_path = Node(current.x_index, current.y_index, current.yaw_index,
                      current.direction, f_x, f_y, f_yaw, f_d,
                      cost=f_cost, parent_index=f_parent_index, steer=f_steer)
        return True, f_path

    return False, None

def calc_motion_inputs():
    for steer in np.concatenate((np.linspace(-MAX_STEER, MAX_STEER,
                                             N_STEER), [0.0])):
        for d in [1, -1]:
            yield [steer, d]

def verify_index(node, c):
    x_ind, y_ind = node.x_index, node.y_index
    if c.min_x_index <= x_ind <= c.max_x_index and c.min_y_index <= y_ind <= c.max_y_index:
        return True
    return False

def get_neighbors(current, config, dist_map):
    for steer, d in calc_motion_inputs():
        node = calc_next_node(current, steer, d, config, dist_map)
        if node and verify_index(node, config):
            yield node

def calc_next_node(current, steer, direction, config, dist_map):
    x, y, yaw = current.x_list[-1], current.y_list[-1], current.yaw_list[-1]

    arc_l = XY_GRID_RESOLUTION
    x_list, y_list, yaw_list, d_list= [], [], [], []
    for _ in np.arange(0, arc_l, MOTION_RESOLUTION):
        x, y, yaw = move(x, y, yaw, MOTION_RESOLUTION * direction, steer)
        x_list.append(x)
        y_list.append(y)
        yaw_list.append(yaw)
        d_list.append(direction)


    if not check_car_collision_dist_map(x_list, y_list, yaw_list, dist_map, config):
        return None

    x_ind = round(x / XY_GRID_RESOLUTION)
    y_ind = round(y / XY_GRID_RESOLUTION)
    yaw_ind = round(yaw / YAW_GRID_RESOLUTION)

    added_cost = 0.0

    if direction != current.direction:
        added_cost += SB_COST

    # steer penalty
    added_cost += STEER_COST * abs(steer)

    # steer change penalty
    added_cost += STEER_CHANGE_COST * abs(current.steer - steer)

    cost = current.cost + added_cost + arc_l

    node = Node(x_ind, y_ind, yaw_ind, direction, x_list,
                y_list, yaw_list, d_list,
                parent_index=calc_index(current, config),
                cost=cost, steer=steer)

    return node

def hybrid_a_star_planning(start, goal, dist_map, xy_resolution, yaw_resolution, obst_grid_resolution):
    """
    start: 起点
    goal: 终点
    dist_map: 障碍物距离地图
    xy_resolution: [m] 网格分辨率
    yaw_resolution: [rad] 偏航角分辨率
    obst_grid_resolution: [m] 障碍物地图分辨率
    """

    start[2], goal[2] = rs.pi_2_pi(start[2]), rs.pi_2_pi(goal[2])

    config = Config(dist_map, xy_resolution, yaw_resolution, obst_grid_resolution)  # 配置地图参数，单位m

    start_node = Node(round(start[0] / xy_resolution),
                      round(start[1] / xy_resolution),
                      round(start[2] / yaw_resolution), True,
                      [start[0]], [start[1]], [start[2]], [True], cost=0)
    goal_node = Node(round(goal[0] / xy_resolution),
                     round(goal[1] / xy_resolution),
                     round(goal[2] / yaw_resolution), True,
                     [goal[0]], [goal[1]], [goal[2]], [True])

    openList, closedList = {}, {}

    pq = [] # [(calc_cost, calc_index)]
    start_node_index = calc_index(start_node, config)
    openList[start_node_index] = start_node
    heapq.heappush(pq, (start_node.cost, start_node_index)) # 最小堆
    final_path = None
    path_list = []
    while True:
        if len(openList) == 0:
            print("Error: Cannot find path, No open set")
            break

        cost, c_id = heapq.heappop(pq)
        if c_id in openList:
            current = openList.pop(c_id)
            closedList[c_id] = current
        else:
            continue

        is_updated, find_path = update_node_with_analytic_expansion(current, goal_node, config, dist_map)

        if is_updated:
            path_list.append(find_path)
            print("path found, cost:", find_path.cost)
            print("path found, num:", len(path_list))
            if show_animation:
                fig, ax = plt.subplots()
                ax.cla()
                ax.imshow(dist_map)
                path = get_final_path(closedList, find_path)
                ax.set_title(str(path.cost))
                ax.plot(np.array(path.x_list)/OBST_GRID_RESOLUTION, np.array(path.y_list)/OBST_GRID_RESOLUTION)
                plt.pause(0.001)
                # plt.waitforbuttonpress()
            if len(path_list) >= MAX_FIND_PATH_NUM:
                break

        # hybrid a star 节点扩展``
        for neighbor in get_neighbors(current, config, dist_map):
            neighbor_index = calc_index(neighbor, config)
            if neighbor_index in closedList:
                continue
            if (neighbor not in openList) or (openList[neighbor_index].cost > neighbor.cost):
                heapq.heappush( pq, (neighbor.cost, neighbor_index))
                openList[neighbor_index] = neighbor
                if show_animation:
                    plt.plot(neighbor.x_list[-1]/config.obst_grid_resolution, neighbor.y_list[-1]/config.obst_grid_resolution, ".")
                    plt.pause(0.001)
                    # plt.waitforbuttonpress()

    final_path = min(path_list, key=lambda o: o.cost)
    print("path found, min cost:", final_path.cost)
    path = get_final_path(closedList, final_path)
    return path


def get_final_path(closed, goal_node):
    reversed_x, reversed_y, reversed_yaw = \
        list(reversed(goal_node.x_list)), list(reversed(goal_node.y_list)), \
        list(reversed(goal_node.yaw_list))
    direction = list(reversed(goal_node.directions))
    nid = goal_node.parent_index
    final_cost = goal_node.cost

    while nid:
        n = closed[nid]
        reversed_x.extend(list(reversed(n.x_list)))
        reversed_y.extend(list(reversed(n.y_list)))
        reversed_yaw.extend(list(reversed(n.yaw_list)))
        direction.extend(list(reversed(n.directions)))

        nid = n.parent_index

    reversed_x = list(reversed(reversed_x))
    reversed_y = list(reversed(reversed_y))
    reversed_yaw = list(reversed(reversed_yaw))
    direction = list(reversed(direction))

    # adjust first direction
    direction[0] = direction[1]

    path = Path(reversed_x, reversed_y, reversed_yaw, direction, final_cost)

    return path

def main():
    img_map, obst_map = get_map(image_path)
    dist_map = get_dist_map(obst_map) * OBST_GRID_RESOLUTION
    
    print("start : ", start)
    print("goal : ", goal)

    if show_animation:
        plt.imshow(dist_map)
        rs.plot_arrow(start[0]/OBST_GRID_RESOLUTION, start[1]/OBST_GRID_RESOLUTION, start[2], length=20, width=20, fc='g')
        rs.plot_arrow(goal[0]/OBST_GRID_RESOLUTION, goal[1]/OBST_GRID_RESOLUTION, goal[2], length=20, width=20)
        plt.grid(True)
        plt.axis("equal")
        plt.pause(0.0001)

    path = hybrid_a_star_planning(start, goal, dist_map, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION, OBST_GRID_RESOLUTION)
    
    x = np.array(path.x_list)
    y = np.array(path.y_list)
    yaw = np.array(path.yaw_list)
    if show_animation_end:
        for i_x, i_y, i_yaw in zip(x, y, yaw):
            plt.cla()
            plt.imshow(img_map)
            plt.plot(x/OBST_GRID_RESOLUTION, y/OBST_GRID_RESOLUTION, "-r", label="Hybrid A* path")
            plt.grid(True)
            plt.axis("equal")
            plot_car(i_x/OBST_GRID_RESOLUTION, i_y/OBST_GRID_RESOLUTION, i_yaw, 1/OBST_GRID_RESOLUTION)
            plt.pause(0.0001)
            # plt.waitforbuttonpress()
        plt.show()
    print(__file__ + " done!!")
    plt.show()

if __name__ == '__main__':
    main()