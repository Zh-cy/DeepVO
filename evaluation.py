import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import os


def euler2mat(euler_path, init_mat_path):
    euler = np.loadtxt(euler_path)
    euler[:, 3:] = euler[:, 3:] * math.pi / 180.
    init_mat = np.loadtxt(init_mat_path)
    dim = len(euler)
    relmat = np.zeros((dim, 4, 4))
    mat = np.zeros(((dim + 1), 4, 4))
    concatenated = np.array([[0., 0., 0., 1.]])
    init_mat = init_mat[0]
    init_mat = np.reshape(init_mat, (3, 4))
    init_mat = np.concatenate((init_mat, concatenated), 0)
    mat[0] = init_mat
    for i in range(dim):
        relmat[i, 0, 0] = math.cos(euler[i, 4]) * math.cos(euler[i, 5])
        relmat[i, 1, 0] = -math.cos(euler[i, 4]) * math.sin(euler[i, 5])
        relmat[i, 2, 0] = math.sin(euler[i, 4])
        relmat[i, 0, 1] = math.sin(euler[i, 3]) * math.sin(euler[i, 4]) * math.cos(euler[i, 5]) + math.cos(euler[i, 3]) * math.sin(euler[i, 5])
        relmat[i, 1, 1] = -math.sin(euler[i, 3]) * math.sin(euler[i, 4]) * math.sin(euler[i, 5]) + math.cos(euler[i, 3]) * math.cos(euler[i, 5])
        relmat[i, 2, 1] = -math.sin(euler[i, 3]) * math.cos(euler[i, 4])
        relmat[i, 0, 2] = -math.cos(euler[i, 3]) * math.sin(euler[i, 4]) * math.cos(euler[i, 5]) + math.sin(euler[i, 3]) * math.sin(euler[i, 5])
        relmat[i, 1, 2] = math.cos(euler[i, 3]) * math.sin(euler[i, 4]) * math.sin(euler[i, 5]) + math.sin(euler[i, 3]) * math.cos(euler[i, 5])
        relmat[i, 2, 2] = math.cos(euler[i, 3]) * math.cos(euler[i, 4])
        relmat[i, 0, 3] = euler[i, 0]
        relmat[i, 1, 3] = euler[i, 1]
        relmat[i, 2, 3] = euler[i, 2]
        relmat[i, 3, 3] = 1.
    for i in range(dim):
        mat[i + 1] = np.dot(mat[i], relmat[i])
    mat = np.resize(mat, (len(mat), 16))
    return mat


if __name__ == '__main__':

    folder = '1_lstm'

    os.mkdir('/path_to_this_folder/KITTI_odometry_evaluation_tool/{}'.format(folder))
    for i in range(11):
        euler_path = '/path_to_prediction_folder/{}/{}.txt'.format(folder, i)
        init_mat_path = 'path_to_this_folder/data/poses/{}.txt'.format(i)
        a = euler2mat(euler_path, init_mat_path)
        b = a[:, :12]
        np.savetxt('path_to_this_folder/KITTI_odometry_evaluation_tool/{}/{:02d}_pred.txt'.
                   format(folder, i), b)
    os.system('python3 /path_to_this_folder/KITTI_odometry_evaluation_tool/evaluation.py '
              '--folder {}'.format(folder))
