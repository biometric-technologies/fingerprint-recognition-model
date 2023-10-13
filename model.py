from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dense, Lambda, Activation, SeparableConv2D, \
    ReLU, GlobalAveragePooling2D
from keras.models import Model
from keras.layers import add
import tensorflow as tf


def create_embedding_model(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), activation='relu')(input_img)
    x = Conv2D(64, (3, 3), activation='relu')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2))(x)
    x = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2))(x)
    x = ReLU()(x)
    x = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(256, (3, 3), padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2))(x)
    x = ReLU()(x)
    x = SeparableConv2D(728, (3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(728, (3, 3), padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = add([x, residual])

    # Middle Flow
    for _ in range(8):
        residual = x
        x = ReLU()(x)
        x = SeparableConv2D(728, (3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(728, (3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(728, (3, 3), padding='same')(x)
        x = add([x, residual])

    # Exit Flow
    residual = Conv2D(1024, (1, 1), strides=(2, 2))(x)
    x = ReLU()(x)
    x = SeparableConv2D(728, (3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = add([x, residual])

    x = SeparableConv2D(1536, (3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(2048, (3, 3), activation='relu', padding='same')(x)

    x = GlobalAveragePooling2D()(x)

    template_output = Dense(96, activation='relu', name='template_output')(x)

    # L2 Normalization
    normalized_template = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name='normalized_template')(
        template_output)

    return Model(inputs=input_img, outputs=normalized_template)
