import numpy as np
from dataset import lstm_test
import argparse
import tensorflow as tf
import os
from model import flownetS_no_weight
from keras import Input, models, layers

parser = argparse.ArgumentParser()
parser.add_argument('--img_height', default=192, type=int, help='image height')
parser.add_argument('--img_width', default=640, type=int, help='image width')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--dropout', default=0, type=float, help='dropout')
parser.add_argument('--recurrent_dropout', default=0, type=float, help='recurrent_dropout')
parser.add_argument('--num_img_pair', default=6, type=int, help='number of images that we need in a sequence')
parser.add_argument('--seq_interval', default=5, type=int, help='the frame that a sequence begins')
parser.add_argument('--folder', type=str, help='folder name to save the training details')
parser.add_argument('--img_folder', default='/mrtstorage/users/czhang/odometry/data_odometry_color/dataset/sequences',
                    type=str, help='the folder that stores images')
parser.add_argument('--cnn_lstm_weights', type=str, help='weights of cnn+lstm')
parser.add_argument('--bidirectional', default=0, type=int, choices=[0, 1], help='0--lstm, 1--bidirectional lstm')

parser.add_argument("--gpu", help='GPU ID list')
parser.add_argument("--works", default=10, type=int, help='Workers number')
args = parser.parse_args()

# variables
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
folder = '/home/path_to_folder_to_save_results/' + args.folder + '_lstm'
lstm_weights = os.path.join(folder, args.cnn_lstm_weights)

# train dataset
save_path = []
for i in range(11):
    save_path.append(os.path.join(folder, '{}.txt'.format(i)))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# model
lstm_shape = (args.num_img_pair - 1, 3, args.img_height, args.img_width)
lstm_i1 = Input(shape=lstm_shape)
lstm_i2 = Input(shape=lstm_shape)
cnn_model = flownetS_no_weight(args.img_height, args.img_width)

l = cnn_model.layers
imgs = layers.concatenate([lstm_i1, lstm_i2], axis=2)
x = layers.TimeDistributed(l[1])(imgs)
for i in range(2, 27):
    x = layers.TimeDistributed(l[i])(x)
x = layers.TimeDistributed(layers.Flatten())(x)
if not args.bidirectional:
    x = layers.LSTM(1000, return_sequences=True, dropout=args.dropout, recurrent_dropout=args.recurrent_dropout)(x)
    x = layers.LSTM(1000, return_sequences=True, dropout=args.dropout, recurrent_dropout=args.recurrent_dropout)(x)
else:
    x = layers.Bidirectional(layers.LSTM(1000, return_sequences=True, dropout=args.dropout,
                                         recurrent_dropout=args.recurrent_dropout))(x)
    x = layers.Bidirectional(layers.LSTM(1000, return_sequences=True, dropout=args.dropout,
                                         recurrent_dropout=args.recurrent_dropout))(x)
    x = layers.Dense(256, activation='relu')(x)
out = layers.Dense(6)(x)
lstm_model = models.Model([lstm_i1, lstm_i2], out)

lstm_model.load_weights(lstm_weights+'.h5')

test_generator = lstm_test(img_folder=args.img_folder,
                           batch_size=args.batch_size,
                           img_height=args.img_height,
                           img_width=args.img_width,
                           num_img_pair=args.num_img_pair,
                           seq_interval=args.seq_interval)

pred = lstm_model.predict_generator(test_generator, steps=test_generator.__len__(), verbose=1)

predicted = [pred[0: 908],
             pred[908: 1128],
             pred[1128: 2060],
             pred[2060: 2220],
             pred[2220: 2274],
             pred[2274: 2826],
             pred[2826: 3046],
             pred[3046: 3266],
             pred[3266: 4080],
             pred[4080: 4398],
             pred[4398: 4638]]


for j in range(len(predicted)):
    predicted[j] = np.reshape(predicted[j], (len(predicted[j])*5, 6))


for i in range(11):
    np.savetxt(save_path[i], predicted[i])
