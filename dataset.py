import cv2
import numpy as np
from keras.utils import Sequence
import math
from glob import glob
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class lstm_train_val(Sequence):
    def __init__(self, img_folder, euler_folder, batch_size, img_height, img_width, num_img_pair, seq_interval, mode):
        self.img_folder = img_folder
        self.euler_folder = euler_folder
        self.batch_size = batch_size
        self.height = img_height
        self.width = img_width
        # number of images in one vider
        # attention: it is not image pairs! Relationship: num_img = num_img_pair - 1)
        self.num_img = num_img_pair  # 3
        self.seq_interval = seq_interval  # 2
        # seq_interval, interval of the first image in every video
        # eg. when num_img = 3：
        # seq_interval = 1 --> video1(0.png, 1.png, 2.png), video2(1.png, 2.png, 3.png)
        # seq_interval = 2 --> video1(0.png, 1.png, 2.png), video2(2.png, 3.png, 4.png)
        # seq_interval = 3 --> video1(0.png, 1.png, 2.png), video2(3.png, 4.png, 5.png)
        # seq_interval = 4 --> video1(0.png, 1.png, 2.png), video2(4.png, 5.png, 6.png)
        self.kitti_mean = [127.5, 127.5, 127.5]
        self.kitti_std = [255, 255, 255]

        if mode == 'train':
            # self.seq = [0, 1, 2, 8, 9]
            self.seq = [0, 1, 2, 4, 6, 8, 10]
            # if u wanna train the model on all sequences with ground truth
            # then u can make a qualitative test on the rest 11 sequences on KITTI
            # self.seq = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
        elif mode == 'val':
            self.seq = [5]
        else:
            self.seq = None
            print("Please give the right mode --> 'train' or 'val' ")

        self.img1_address, self.img2_address, self.pose = self.load_data()
        if mode == 'train' and self.seq == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]:
            self.img1_address = self.img1_address[:4084]
            self.img2_address = self.img2_address[:4084]
            self.pose = self.pose[:4084]

    def __len__(self):
        return math.ceil(len(self.img1_address) / self.batch_size)

    def __getitem__(self, idx):
        # return value should be [(bs, num_img-1, 3, 384, 1280), (bs, num_img-1, 3, 384, 1280)], (bs, num_img-1, 6)
        img1 = self.img1_address[idx * self.batch_size: (idx + 1) * self.batch_size]
        img2 = self.img2_address[idx * self.batch_size: (idx + 1) * self.batch_size]
        y_true = self.pose[idx * self.batch_size: (idx + 1) * self.batch_size]
        img1_arrays = np.empty((self.batch_size, self.num_img - 1, 3, self.height, self.width))
        img2_arrays = np.empty((self.batch_size, self.num_img - 1, 3, self.height, self.width))
        pose_array = np.reshape(y_true, (self.batch_size, self.num_img - 1, 6))
        for i in range(self.batch_size):
            for j in range(self.num_img - 1):
                img1_arrays[i, j, :, :, :] = self.img_get(img1[i, j])
                img2_arrays[i, j, :, :, :] = self.img_get(img2[i, j])
        return [img1_arrays, img2_arrays], pose_array

    def load_data(self):
        # load data, return 3 array
        # but there is a temporal array, every row is a video, number of cols is num_img
        # this temporal array will be used to generate the other two array
        # first array: (batches, num_img-1), every row is a video, number of cols is num_img-1  --> self.img1_address
        # secont array: (batches, num_img-1), every row is a video
        # every image in this array is the next frame corresponding to the first array  --> self.img1_address
        # third array: (batches, (num_img-1)*6), relative poses of images in the first and second array  --> self.pose
        all_imgs = np.empty((1, self.num_img), dtype=object)
        euler_array = np.empty((1, (self.num_img - 1) * 6), dtype=float)
        for i in self.seq:
            imgs_list = []
            euler_list = []
            seq_start_img = []
            imgs = glob(self.img_folder + '/{:02d}/image_2/*.png'.format(i))
            euler = np.loadtxt(self.euler_folder + '/{}.txt'.format(i))
            imgs.sort()
            imgs_list.extend(imgs)
            euler_list.extend(euler)
            euler_list = np.array(euler_list)

            for j in range(len(imgs_list)):
                if j <= len(imgs_list) - self.num_img and j % self.seq_interval == 0:
                    seq_start_img.append(j)
            to_save = np.empty((len(seq_start_img), self.num_img), dtype=object)
            euler_to_save = np.empty((len(seq_start_img), (self.num_img - 1) * 6))

            for k in range(len(seq_start_img)):
                count = 0
                while count < self.num_img:
                    to_save[k, count] = imgs_list[seq_start_img[k] + count]
                    count += 1
                count = 0
                while count < self.num_img - 1:
                    euler_to_save[k, count * 6: (count + 1) * 6] = euler_list[seq_start_img[k] + count]
                    count += 1

            all_imgs = np.concatenate((all_imgs, to_save), axis=0)
            all_imgs = np.array(all_imgs)
            euler_array = np.concatenate((euler_array, euler_to_save), axis=0)
            euler_array = np.array(euler_array)

        all_imgs = all_imgs[1:]
        img1_array = all_imgs[:, :-1]
        img2_array = all_imgs[:, 1:]
        euler_array = euler_array[1:]

        return img1_array, img2_array, euler_array

    def img_get(self, img_path):
        img = np.array(Image.open(img_path).resize((self.width, self.height)))
        img = (img - self.kitti_mean) / self.kitti_std
        img = np.transpose(img, [2, 0, 1])  # use the flownet checkpoint from pytorch so channel is the first parameter
        # img = img[[2, 1, 0], :, :]  # RGB-->BGR
        return img


class lstm_test(Sequence):
    def __init__(self, img_folder, batch_size, img_height, img_width, num_img_pair, seq_interval):

        self.img_folder = img_folder
        self.batch_size = batch_size
        self.height = img_height  # 384
        self.width = img_width  # 1280
        self.num_img = num_img_pair  # 3
        self.seq_interval = seq_interval  # 2
        # default: test on first 11 sequences on KITTI
        # u can also expand it on all 22 sequences on KITTI
        self.seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.kitti_mean = [127.5, 127.5, 127.5]
        self.kitti_std = [255, 255, 255]
        self.img1_address, self.img2_address = self.load_data()

    def __len__(self):
        return math.ceil(len(self.img1_address) / self.batch_size)

    def __getitem__(self, idx):
        img1 = self.img1_address[idx * self.batch_size: (idx + 1) * self.batch_size]
        img2 = self.img2_address[idx * self.batch_size: (idx + 1) * self.batch_size]
        img1_arrays = np.empty((self.batch_size, self.num_img - 1, 3, self.height, self.width))
        img2_arrays = np.empty((self.batch_size, self.num_img - 1, 3, self.height, self.width))
        for i in range(self.batch_size):
            for j in range(self.num_img - 1):
                img1_arrays[i, j, :, :, :] = self.img_get(img1[i, j])
                img2_arrays[i, j, :, :, :] = self.img_get(img2[i, j])
        return [img1_arrays, img2_arrays]

    def load_data(self):
        all_imgs = np.empty((1, self.num_img), dtype=object)
        for i in self.seq:
            imgs_list = []
            seq_start_img = []
            imgs = glob(self.img_folder + '/{:02d}/image_2/*.png'.format(i))
            imgs.sort()
            imgs_list.extend(imgs)
            for j in range(len(imgs_list)):
                if j <= len(imgs_list) - self.num_img and j % self.seq_interval == 0:
                    seq_start_img.append(j)
            to_save = np.empty((len(seq_start_img), self.num_img), dtype=object)

            for k in range(len(seq_start_img)):
                count = 0
                while count < self.num_img:
                    to_save[k, count] = imgs_list[seq_start_img[k] + count]
                    count += 1

            all_imgs = np.concatenate((all_imgs, to_save), axis=0)
            all_imgs = np.array(all_imgs)
        all_imgs = all_imgs[1:]
        img1_array = all_imgs[:, :-1]
        img2_array = all_imgs[:, 1:]

        return img1_array, img2_array

    def img_get(self, img_path):
        img = np.array(Image.open(img_path).resize((self.width, self.height)))
        img = (img - self.kitti_mean) / self.kitti_std
        img = np.transpose(img, [2, 0, 1])
        # img = img[[2, 1, 0], :, :]  # RGB-->BGR
        return img


