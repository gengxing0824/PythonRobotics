import numpy as np
import matplotlib.pyplot as plt
from get_map import get_map

def get_min_dist_point_to_point(dis_nums, row_nums, col_nums, row_org, col_org,  row_target, col_target):
    row_obst = row_nums[row_target, col_target]
    col_obst = col_nums[row_target, col_target]
    dis_temp = np.hypot(row_org - row_obst, col_org - col_obst)

    if dis_temp < dis_nums[row_org][col_org]:
        dis_nums[row_org, col_org] = dis_temp
        row_nums[row_org, col_org] = row_obst
        col_nums[row_org, col_org] = col_obst
    return 


def get_dist_froward(dis_nums, row_nums, col_nums, row, col):
    if row == 0 or col == 0 or row == len(dis_nums) - 1 or col == len(dis_nums[0]) - 1 or dis_nums[row, col] == 0:
        row_nums[row, col] = row
        col_nums[row, col] = col
        dis_nums[row, col] = 0
        return
    
    get_min_dist_point_to_point(dis_nums, row_nums, col_nums, row, col, row, col-1)
    # get_min_dist_point_to_point(dis_nums, row_nums, col_nums, row, col, row-1, col-1)
    # get_min_dist_point_to_point(dis_nums, row_nums, col_nums, row, col, row-1, col)
    get_min_dist_point_to_point(dis_nums, row_nums, col_nums, row, col, row-1, col+1)
    return 

def get_dist_backward(dis_nums, row_nums, col_nums, row, col):
    if row == 0 or col == 0 or row == len(dis_nums) - 1 or col == len(dis_nums[0]) - 1 or dis_nums[row, col] == 0:
        row_nums[row, col] = row
        col_nums[row, col] = col
        dis_nums[row, col] = 0
        return
    get_min_dist_point_to_point(dis_nums, row_nums, col_nums, row, col, row, col+1)
    # get_min_dist_point_to_point(dis_nums, row_nums, col_nums, row, col, row+1, col+1)
    # get_min_dist_point_to_point(dis_nums, row_nums, col_nums, row, col, row+1, col)
    get_min_dist_point_to_point(dis_nums, row_nums, col_nums, row, col, row+1, col-1)
    return 
    
def get_dist_map(nums):
    row_nums = -1 * np.ones_like(nums)
    col_nums = -1 * np.ones_like(nums)
    dis_nums = nums * 10000
    for row in range(nums.shape[0]):
        for col in range(nums.shape[1]):
            get_dist_froward(dis_nums, row_nums, col_nums, row, col)
    for row in range(nums.shape[0]-1, -1, -1):
        for col in range(nums.shape[1]-1, -1, -1):
            get_dist_backward(dis_nums, row_nums, col_nums, row, col)
    return dis_nums


if __name__ == "__main__":
    image_path = "MyLearning\\HybridAStar\\fusion_map_2023-03-19-04-11-11.jpg"
    t, obst_nums = get_map(image_path)
    plt.imshow(t)
    plt.show()
    plt.imshow(obst_nums)
    plt.show()
    dis_nums = get_dist_map(obst_nums) * 0.05
    plt.imshow(dis_nums)
    # plt.imshow(nums)
    plt.show()

