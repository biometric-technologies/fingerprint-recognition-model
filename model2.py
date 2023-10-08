from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dense, Lambda, GlobalAveragePooling2D, \
    BatchNormalization, ReLU
from keras.models import Model
import tensorflow as tf


def add_resblock(x, num_filters, num_conv):
    shortcut = x
    for _ in range(num_conv):
        x = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    x = Add()([shortcut, x])
    return x


def create_embedding_model(input_shape):
    input_img = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(input_img)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = add_resblock(x, 64, 3)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = add_resblock(x, 128, 4)
    x = add_resblock(x, 256, 6)
    x = add_resblock(x, 512, 3)

    x = GlobalAveragePooling2D()(x)

    template_output = Dense(128, activation='relu', name='template_output')(x)
    # L2 Normalization
    normalized_template = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name='normalized_template')(
        template_output)

    return Model(inputs=input_img, outputs=normalized_template)
