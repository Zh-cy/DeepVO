import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def loss_plot(path_val, path_train, save_path):
    val = pd.read_csv(path_val, header=None, sep=',')
    train = pd.read_csv(path_train, header=None, sep=',')
    val = np.array(val[1:], dtype='float64')
    train = np.array(train[1:], dtype='float64')

    x = val[:, 1]
    val_loss = val[:, 2]
    train_loss = train[:, 2]
    plt.xlabel('Epoch', fontsize=15, fontdict={'family': 'Times New Roman'})
    plt.ylabel('Loss', fontsize=15, fontdict={'family': 'Times New Roman'})
    plt.plot(x, train_loss, color='blue', label='Train Loss')
    plt.plot(x, val_loss, color='red', label='Validation Loss')
    plt.legend(loc='upper right', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(xmin=0)
    plt.xlim(xmax=101)
    plt.ylim(ymin=0)
    plt.ylim(ymax=4)
    plt.savefig(save_path)
    plt.show()


def loss_arr(path_val, path_train):
    val = pd.read_csv(path_val, header=None, sep=',')
    train = pd.read_csv(path_train, header=None, sep=',')
    val = np.array(val[1:], dtype='float64')
    train = np.array(train[1:], dtype='float64')
    train_loss = train[:, 2]
    val_loss = val[:, 2]
    return train_loss, val_loss


if __name__ == "__main__":
    folder = '1_lstm'
    path_train = '/home/users/lstm_vo/{}/run-20200919-174239lstm-tag-loss.csv'.format(folder)
    path_val = '/mrtstorage/users/czhang/lstm_vo/22_lstm!!!/test3/run-20200919-174239lstm-tag-val_loss.csv'
    save_path = '/mrtstorage/users/czhang/lstm_vo/{}/loss.jpg'.format(folder)
    loss_plot(path_val, path_train, save_path)
