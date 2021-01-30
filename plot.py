import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D


def relmat2euler(relmat_path):
    """
    left_hand!
    R = Rz*Ry*Rx!!!
    thetax = atan2(-r32,r33)
    thetay = atan2(r31, math.sqrt(r32**2 + r33**2))
    thetaz = atan2(-r21, r11)
    :param mat_path: path to rotation matrix of a sequence
    :return: array to store euler angles of a sequence
    """
    relmat_seq = np.loadtxt(relmat_path)
    dim = len(relmat_seq)
    euler_seq = np.zeros((dim, 6))
    for i in range(len(relmat_seq)):
        mat = relmat_seq[i]
        euler = euler_seq[i]
        euler[0] = mat[3]
        euler[1] = mat[7]
        euler[2] = mat[11]
        euler[3] = math.atan2(-mat[9], mat[10])  # thetax
        euler[4] = math.atan2(mat[8], math.sqrt(mat[9]**2. + mat[10]**2.))  # thetay
        euler[5] = math.atan2(-mat[4], mat[0])  # thetaz
    return euler_seq


def mat2relmat(mat_path):
    """
    T1 = 1T0 dot T0 ---->>>> Wrong!!!
    1T0 = T1 dot inverse(T0) ---->>>> Wrong!!!
    It is relative to a dynamic reference system!!!!!
    T1 = T0 dot delta(T)
    delta(T) = inverse(T0) dot T1
    :param mat_path:
    :return: relative matrix between 2 images
    """
    # absolute pose(dim, 12)
    mat_seq = np.loadtxt(mat_path)
    dim = len(mat_seq)
    # relative pose (dim-1, 4, 4)
    relmat = np.zeros((dim-1, 4, 4))
    # absolute pose --> (dim, 3, 4)
    mat_seq = np.reshape(mat_seq, (dim, 3, 4))
    concatenated = np.zeros((dim, 1, 4))
    concatenated[:, 0, 3] = 1.
    mat_seq = np.concatenate((mat_seq, concatenated), 1)
    for i in range(dim-1):
        inv = np.linalg.inv(mat_seq[i])
        relmat[i] = np.dot(inv, mat_seq[i+1])
    relmat = np.reshape(relmat, (dim-1, 16))
    return relmat


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


def save_pose_mat(euler_path, init_mat_path, save_mat_path):
    mat_16 = euler2mat(euler_path, init_mat_path)
    mat_16 = np.reshape(mat_16, (len(mat_16), 4, 4))
    mat_12 = mat_16[:, :3, :]
    mat_12 = np.reshape(mat_12, (len(mat_12), 12))
    np.savetxt(save_mat_path, mat_12)


def euler2plot(pred_euler_path, gt_euler_path, init_mat_path, save_path, mode_2d=True):
    pred_mat = euler2mat(pred_euler_path, init_mat_path)
    gt_mat = euler2mat(gt_euler_path, init_mat_path)
    x_pred = pred_mat[:, 3]
    y_pred = pred_mat[:, 7]
    z_pred = pred_mat[:, 11]
    x_gt = gt_mat[:, 3]
    y_gt = gt_mat[:, 7]
    z_gt = gt_mat[:, 11]

    if mode_2d:
        fig = plt.figure()
        figure = plt.plot(x_gt, z_gt, color='blue', label='Ground Truth')
        figure = plt.plot(x_pred, z_pred, color='red', label='Experiment b')
        plt.legend(loc='upper right')
        plt.savefig(save_path)
    else:
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.set_title('3D_Curve')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        figure = ax.plot(x_gt, y_gt, z_gt, color='blue', label='ground_truth')
        figure = plt.plot(x_pred, y_pred, z_pred, color='red', label='ours')
        plt.legend(loc='upper right')
        plt.savefig(save_path)


def euler2plot_no_gt(pred_euler_path, save_path, mode_2d=True):
    init_mat_path = '/home/user/workspaces/test_ws/src/DeepVO/data/poses/0.txt'
    pred_mat = euler2mat(pred_euler_path, init_mat_path)
    x_pred = pred_mat[:, 3]
    y_pred = pred_mat[:, 7]
    z_pred = pred_mat[:, 11]

    if mode_2d:
        fig = plt.figure()
        figure = plt.plot(x_pred, z_pred, color='red', label='CCL-VO')
        plt.legend(loc='upper right')
        plt.savefig(save_path)
    else:
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.set_title('3D_Curve')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        figure = plt.plot(x_pred, y_pred, z_pred, color='red', label='ours')
        plt.legend(loc='upper right')
        plt.savefig(save_path)


if __name__ == "__main__":

    folder = '30_lstm'
    for i in range(11, 22):
        predicted = '/home/user/workspaces/test_ws/src/test/{}/{}.txt'.format(folder, i)
        save = '/home/user/workspaces/test_ws/src/test/{}/{}.jpg'.format(folder, i)
        euler2plot_no_gt(predicted, save)
        plt.show()
