from keras import layers, Input, models
from keras.initializers import Constant
from keras import backend as K


def conv_bn(input_tensor, num_out, kernal_size, strides, padding='same', initializer='glorot_normal', batch_norm=False):
    y = layers.Conv2D(filters=num_out,
                      kernel_size=kernal_size,
                      strides=strides,
                      padding=padding,
                      kernel_initializer=initializer)(input_tensor)
    if batch_norm:
        y = layers.BatchNormalization()(y)
        output = layers.Activation('relu')(y)
    else:
        output = layers.Activation('relu')(y)
    return output


def dense_bn(input_tensor, num_out, initializer='glorot_normal', batch_norm=False):
    y = layers.Dense(num_out, kernel_initializer=initializer)(input_tensor)
    if batch_norm:
        y = layers.BatchNormalization()(y)
        output = layers.Activation('relu')(y)
    else:
        output = layers.Activation('relu')(y)
    return output


def squeeze_excite_block(tensor, ratio=16):
    """
    from: https://github.com/titu1994/keras-squeeze-excite-network
    References: [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1  # channel层在第一层还是最后一层(c, h, w) or (h, w, c)
    filters = init._keras_shape[channel_axis]  # filter层数
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = layers.Permute((3, 1, 2))(se)

    x = layers.multiply([init, se])
    return x


def flownetS_to_load(height, width):
    K.set_image_data_format('channels_first')
    pre_trained_weights_path = '/home/users/path_to_this_weights/pre_trained_flownetS_keras.h5'
    imgs = Input(shape=(6, height, width))
    x = layers.ZeroPadding2D((3, 3))(imgs)
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((2, 2))(x)
    x = layers.Conv2D(filters=128, kernel_size=5, strides=2)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((2, 2))(x)
    x = layers.Conv2D(filters=256, kernel_size=5, strides=2)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=2)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=2)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Conv2D(filters=1024, kernel_size=3, strides=2)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Conv2D(filters=1024, kernel_size=3, strides=1)(x)
    out = layers.Activation('relu')(x)
    model = models.Model(imgs, out)
    model.load_weights(pre_trained_weights_path)
    return model


def flownetS_no_weight(height, width):
    K.set_image_data_format('channels_first')
    imgs = Input(shape=(6, height, width))
    x = layers.ZeroPadding2D((3, 3))(imgs)
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((2, 2))(x)
    x = layers.Conv2D(filters=128, kernel_size=5, strides=2)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((2, 2))(x)
    x = layers.Conv2D(filters=256, kernel_size=5, strides=2)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=2)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=2)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Conv2D(filters=1024, kernel_size=3, strides=2)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Conv2D(filters=1024, kernel_size=3, strides=1)(x)
    out = layers.Activation('relu')(x)
    model = models.Model(imgs, out)
    return model


def cnn_lstm(input1, input2, dropout=False):
    imgs = layers.concatenate([input1, input2], axis=-1)
    x = layers.TimeDistributed(layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                                             kernel_initializer='glorot_normal', activation='relu'))(imgs)
    x = layers.TimeDistributed(layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                             kernel_initializer='glorot_normal', activation='relu'))(x)
    x = layers.TimeDistributed(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                             kernel_initializer='glorot_normal', activation='relu'))(x)
    x = layers.TimeDistributed(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             kernel_initializer='glorot_normal', activation='relu'))(x)
    x = layers.TimeDistributed(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                             kernel_initializer='glorot_normal', activation='relu'))(x)
    x = layers.TimeDistributed(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             kernel_initializer='glorot_normal', activation='relu'))(x)
    x = layers.TimeDistributed(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                             kernel_initializer='glorot_normal', activation='relu'))(x)
    x = layers.TimeDistributed(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             kernel_initializer='glorot_normal', activation='relu'))(x)
    x = layers.TimeDistributed(layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                             kernel_initializer='glorot_normal', activation='relu'))(x)
    x = layers.TimeDistributed(layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             kernel_initializer='glorot_normal'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2), strides=(2, 2)))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.LSTM(1024, return_sequences=True)(x)
    x = layers.LSTM(1024, return_sequences=True)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(6)(x)
    model = models.Model([input1, input2], out)
    return model
