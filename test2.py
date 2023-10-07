import tensorflow as tf
from keras.models import Model, load_model
import cv2
import numpy as np


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (192, 192))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # Convert to shape (192, 192, 1)
    return np.expand_dims(img, axis=0)


if __name__ == '__main__':
    saved_model_path = "saved_model2"
    loaded_model = load_model(saved_model_path)
    # Extract the embedding model (assuming embedding model is the first part of your triplet model)
    embedding_model = Model(inputs=loaded_model.input[0], outputs=loaded_model.layers[-3].output)
    image1_path = 'image1.jpg'
    image2_path = 'image2.jpg'
    image1 = preprocess_image(image1_path)
    image2 = preprocess_image(image2_path)
    embedding1 = embedding_model.predict(image1)
    embedding2 = embedding_model.predict(image2)
    l1_distance = np.sum(np.abs(embedding1 - embedding2))
    print(f"L1 Distance between images: {l1_distance}")
