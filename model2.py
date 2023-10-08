from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dense, Lambda, GlobalAveragePooling2D, \
    BatchNormalization, ReLU
from keras.models import Model
import tensorflow as tf

def res_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    # First convolution
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Second convolution
    x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    # Adjusting the shortcut for addition if necessary
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)
    # Adding the shortcut to the output
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x


def create_embedding_model(input_shape):
    input_img = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Residual Blocks
    x = res_block(x, 64)
    x = res_block(x, 64)
    x = res_block(x, 128, stride=2)  # Downsample
    x = res_block(x, 128)
    x = res_block(x, 256, stride=2)  # Downsample
    x = res_block(x, 256)
    x = res_block(x, 512, stride=2)  # Downsample
    x = res_block(x, 512)

    x = GlobalAveragePooling2D()(x)

    template_output = Dense(128, activation='relu', name='template_output')(x)
    # L2 Normalization
    normalized_template = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name='normalized_template')(
        template_output)

    return Model(inputs=input_img, outputs=normalized_template)
