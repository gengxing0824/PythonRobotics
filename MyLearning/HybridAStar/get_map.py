import matplotlib.pyplot as plt
import numpy as np

def get_map(image_path):
    image = plt.imread(image_path)
    obst_nums = np.sum(image, axis = 2) > 100
    if (obst_nums.shape[0] > obst_nums.shape[1]):
        image = np.rot90(image)
        obst_nums = np.rot90(obst_nums)
    return image, obst_nums

if __name__ == "__main__":
    image_path = "MyLearning\\GetMap\\fusion_map_2023-03-19-04-11-11.jpg"
    obst_nums = get_map(image_path)
    print(obst_nums.shape)
    plt.imshow(obst_nums)
    plt.show()