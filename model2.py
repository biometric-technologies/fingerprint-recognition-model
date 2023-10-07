from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from keras.models import Model
import tensorflow as tf


def create_unet_model(input_shape):
    input_img = Input(shape=(192, 192, 1))

    # Encoding/Downsampling
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Flattening and creating a fixed-size vector
    flat = Flatten()(encoded)
    dense1 = Dense(512, activation='relu')(flat)
    dense2 = Dense(256, activation='relu')(dense1)
    template_output = Dense(128, activation='relu', name='template_output')(dense2)

    # L2 Normalization
    normalized_template = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name='normalized_template')(
        template_output)

    return Model(inputs=input_img, outputs=normalized_template)
