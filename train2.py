import os
import cv2
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.models import Model, save_model
import model2
from tripplet_generator import TripletGenerator


def triplet_loss(y_true, y_pred, margin=1.0):
    anchor, positive, negative = y_pred[:, :128], y_pred[:, 128:256], y_pred[:, 256:]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
    return tf.reduce_mean(loss)


def load_and_preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (192, 192))
    img = img / 255.0
    return img


if __name__ == '__main__':
    root_folder = 'fingerprint-v5-master'

    embedding_model = model2.create_unet_model((192, 192, 1))
    print(embedding_model.summary())

    input_anchor = Input(shape=(192, 192, 1))
    input_positive = Input(shape=(192, 192, 1))
    input_negative = Input(shape=(192, 192, 1))

    embedding_anchor = embedding_model(input_anchor)
    embedding_positive = embedding_model(input_positive)
    embedding_negative = embedding_model(input_negative)

    output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)
    triplet_model = Model([input_anchor, input_positive, input_negative], output)
    triplet_model.compile(optimizer='adam', loss=triplet_loss)

    checkpoint = ModelCheckpoint('model_weights_best.h5', save_best_only=True, monitor='loss')

    triplet_gen = TripletGenerator(root_folder, 32)
    triplet_model.fit(triplet_gen, epochs=64)

    saved_model_path = os.path.join(".", 'saved_model')
    save_model(triplet_model, saved_model_path)
