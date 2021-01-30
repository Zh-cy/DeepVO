import smtplib
import ssl
import argparse
import keras
from keras import Input, models, layers
from dataset import lstm_train_val
from keras import backend as K
from datetime import datetime
import tensorflow as tf
import os
import math
import model
from keras.callbacks import LearningRateScheduler
from keras.utils import multi_gpu_model

parser = argparse.ArgumentParser()
# model
parser.add_argument('--initial_epoch', default=0, type=int, help='initial epoch')
parser.add_argument('--cnn_lstm_weights', type=str, help='weights of cnn+lstm')
# like this default='cnn_lstm_weight.69-0.7205.h5'
parser.add_argument('--pre_trained', default=1, type=int, choices=[0, 1],
                    help='1--load cnn pre-trained weights to train, 0--train cnn_lstm model from the beginning')
parser.add_argument('--resume', default=0, type=int, choices=[0, 1],
                    help='0--train from beginning, 1--load checkpoint to train')
parser.add_argument('--bidirectional', default=0, type=int, choices=[0, 1], help='0--lstm, 1--bidirectional lstm')

parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
parser.add_argument('--num_img_pair', default=6, type=int, help='number of images that we need in a sequence')
parser.add_argument('--seq_interval', default=5, type=int, help='the frame that a sequence begins')
# eg: num_img_pair = 3; seq_interval = 2  --> seq1:(0, 1, 2); seq2:(2, 3, 4); seq3(4, 5, 6)...
parser.add_argument('--epoch_max', default=200, type=int, help='Max epoch')
parser.add_argument('--img_height', default=192, type=int, help='image height')
parser.add_argument('--img_width', default=640, type=int, help='image width')
parser.add_argument('--batch_norm', default=0, type=int, choices=[0, 1],
                    help='if we use the batch normalization, 0:False, 1:True')
parser.add_argument('--dropout', default=0, type=float, help='dropout')
parser.add_argument('--recurrent_dropout', default=0, type=float, help='recurrent_dropout')
parser.add_argument('--save_model_interval', default=1, type=int, help='every how many epochs to store the model')
parser.add_argument('--tensor_board_log', default='/home/user/pathxxxxx/logs_lstm/',
                    help='tensor board log save path')
parser.add_argument('--lr_strategy', default='step', help='learning rate strategy')
parser.add_argument('--lr_base', default=1e-4, type=float, help='Base learning rate')
parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='Decay rate of lr')
parser.add_argument('--epoch_lr_decay', type=int, help='Every # epoch, lr decay lr_decay_rate')
# didn't use this strategy, just keep the learning rate value
parser.add_argument('--beta', default=100, type=int, help='loss = loss_t + beta * loss_r')
# this is an important parameter, Loss = Lt + beta * Lr
# set it in range [1, 100]
parser.add_argument('--folder', type=str, help='folder name to save the training details')

parser.add_argument("--gpu", help='GPU id list')
parser.add_argument("--workers", default=10, type=int, help='Workers number')
# server
parser.add_argument('--img_folder', default='/home/zhangsan/odometry/data_odometry_color/dataset/sequences',
                    type=str, help='the folder that stores images')
parser.add_argument('--euler_folder', default='/path_to_euler_angle/euler_angle',
                    type=str, help='the folder that stores euler angle files')
# email
# u can send the training information to your smartphone per Email
parser.add_argument('--smtp_server', default="smtp.gmail.com", type=str)
parser.add_argument('--port', default=465, type=int)
parser.add_argument('--sender_email', default="xxxxxx@xxx.com", type=str)
parser.add_argument('--receiver_email', default="xxxxxx@xxx.com", type=str)
parser.add_argument('--password', default="hnzymhlxylhtpdtt", type=str)
args = parser.parse_args()

# variables
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

folder = '/mrtstorage/users/DeepVO/' + args.folder + '_lstm'

# checkpoint of cnn_lstm, this is to save the weights of cnn_lstm
checkpoint_path = os.path.join(folder, 'cnn_lstm_weight.{epoch:02d}-{val_loss:.4f}.h5')

# cnn_lstm weights, if the training process falls due to some reason, I can load this weights to continue my training
lstm_weights = os.path.join(folder, args.cnn_lstm_weights)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


class EmailSender(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # email setup
        port = args.port  # For SSL
        smtp_server = args.smtp_server
        sender_email = args.sender_email  # Enter your address
        receiver_email = args.receiver_email  # Enter receiver address
        password = args.password
        message = str(epoch) + "\n" + str(logs.get('train_loss')) + "\n" + str(logs.get('val_loss')) \
                  + "\n" + str(logs.get('lr'))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)


def lr_scheduler(epoch, mode='step'):
    if mode is 'step':
        lr = args.lr_base * (args.lr_decay_rate ** math.ceil(epoch / args.epoch_lr_decay))
    elif mode is 'fixed':
        lr = args.lr_base
    else:
        raise TypeError('Please give the learning rate scheduler strategy!')
    return lr


# callbacks
earlystop = keras.callbacks.EarlyStopping(monitor='mean_absolute_error', patience=1)
checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                             monitor='val_loss',
                                             save_weights_only=True,
                                             verbose=1,
                                             save_best_only=False,
                                             mode='min',
                                             period=args.save_model_interval)
tensorboard = keras.callbacks.TensorBoard(log_dir=args.tensor_board_log +
                                                  datetime.now().strftime("%Y%m%d-%H%M%S") + str("lstm"),
                                          histogram_freq=0)
if args.lr_strategy is 'reduce':
    scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.lr_decay_rate, patience=10)
elif args.lr_strategy is 'step':
    scheduler = LearningRateScheduler(lr_scheduler)
else:
    raise TypeError('learning rate strategy must be reduce or step')
email_sender = EmailSender()

# callbacks_list = [checkpoint, scheduler, tensorboard, email_sender]
callbacks_list = [checkpoint, scheduler, tensorboard]
# model
lstm_shape = (args.num_img_pair - 1, 3, args.img_height, args.img_width)
lstm_i1 = Input(shape=lstm_shape)
lstm_i2 = Input(shape=lstm_shape)

if not args.resume:
    if args.pre_trained:
        cnn_model = model.flownetS_to_load(args.img_height, args.img_width)
        l = cnn_model.layers
        imgs = layers.concatenate([lstm_i1, lstm_i2], axis=2)
        x = layers.TimeDistributed(l[1])(imgs)
        for i in range(2, 27):
            x = layers.TimeDistributed(l[i])(x)
        x = layers.TimeDistributed(layers.Flatten())(x)
        if not args.bidirectional:
            x = layers.LSTM(1000, return_sequences=True, dropout=args.dropout,
                            recurrent_dropout=args.recurrent_dropout)(x)
            x = layers.LSTM(1000, return_sequences=True, dropout=args.dropout,
                            recurrent_dropout=args.recurrent_dropout)(x)
        else:
            x = layers.Bidirectional(layers.LSTM(1000, return_sequences=True, dropout=args.dropout,
                                                 recurrent_dropout=args.recurrent_dropout))(x)
            x = layers.Bidirectional(layers.LSTM(1000, return_sequences=True, dropout=args.dropout,
                                                 recurrent_dropout=args.recurrent_dropout))(x)
            x = layers.Dense(256, activation='relu')(x)
        out = layers.Dense(6)(x)
        lstm_model = models.Model([lstm_i1, lstm_i2], out)
    else:
        lstm_model = model.cnn_lstm(lstm_i1, lstm_i2)
else:
    cnn_model = model.flownetS_no_weight(args.img_height, args.img_width)
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
    lstm_model.load_weights(lstm_weights)
    print('checkpoint_loaded')


def my_loss(y_true, y_pred):
    beta = args.beta
    y_true = K.reshape(y_true, (-1, 6))
    y_pred = K.reshape(y_pred, (-1, 6))
    t_true = y_true[:, :3]  # [bs, 3]
    r_true = y_true[:, 3:]  # [bs, 3]
    t_pred = y_pred[:, :3]  # [bs, 3]
    r_pred = y_pred[:, 3:]  # [bs, 3]
    loss_t = K.mean((t_pred - t_true) ** 2., -1)  # [bs, ]
    loss_r = K.mean((r_pred - r_true) ** 2., -1)  # [bs, ]
    loss = loss_t + beta * loss_r  # [bs, ]
    return K.mean(loss)


optimizer = keras.optimizers.Adam(lr=args.lr_base)
if len(args.gpu) > 1:
    lstm_model = multi_gpu_model(lstm_model, gpus=len(args.gpu.split(',')))
lstm_model.compile(optimizer=optimizer, loss=my_loss, metrics=['mae'])

train_generator = lstm_train_val(img_folder=args.img_folder,
                                 euler_folder=args.euler_folder,
                                 batch_size=args.batch_size,
                                 img_height=args.img_height,
                                 img_width=args.img_width,
                                 num_img_pair=args.num_img_pair,
                                 seq_interval=args.seq_interval,
                                 mode='train')
val_generator = lstm_train_val(img_folder=args.img_folder,
                               euler_folder=args.euler_folder,
                               batch_size=args.batch_size,
                               img_height=args.img_height,
                               img_width=args.img_width,
                               num_img_pair=args.num_img_pair,
                               seq_interval=args.seq_interval,
                               mode='val')
history = lstm_model.fit_generator(train_generator,
                                   steps_per_epoch=train_generator.__len__(),
                                   epochs=args.epoch_max,
                                   callbacks=callbacks_list,
                                   validation_data=val_generator,
                                   validation_steps=val_generator.__len__(),
                                   initial_epoch=args.initial_epoch,
                                   use_multiprocessing=True,
                                   workers=args.workers)
