from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Softmax


def create_unet_model(input_shape):
    input_img = Input(input_shape)

    # Encoding/Downsampling
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoding/Upsampling for classification
    x_class = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x_class = UpSampling2D((2, 2))(x_class)
    x_class = Conv2D(64, (3, 3), activation='relu', padding='same')(x_class)
    x_class = UpSampling2D((2, 2))(x_class)
    x_class = Conv2D(3, (3, 3), activation='softmax', padding='same', name="classification")(
        x_class)  # 3 channels for classes

    # Decoding/Upsampling for orientation angle prediction
    x_regress = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x_regress = UpSampling2D((2, 2))(x_regress)
    x_regress = Conv2D(64, (3, 3), activation='relu', padding='same')(x_regress)
    x_regress = UpSampling2D((2, 2))(x_regress)
    x_regress = Conv2D(1, (3, 3), activation='linear', padding='same', name="orientation")(
        x_regress)  # 1 channel for angle

    # Create Model
    return Model(inputs=input_img, outputs=[x_class, x_regress])
