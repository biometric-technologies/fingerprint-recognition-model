import os
import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Input
from keras.models import Model, save_model

import model2


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

    # Lists to store data
    anchors = []
    positives = []
    negatives = []

    individuals = os.listdir(root_folder)

    for i in range(len(individuals)):
        current_individual = individuals[i]
        for hand in ['L', 'R']:
            anchor_path = os.path.join(root_folder, current_individual, hand)
            positive_hand = 'R' if hand == 'L' else 'L'
            positive_path = os.path.join(root_folder, current_individual, positive_hand)

            # Load anchor and positive images
            for anchor_img_name in os.listdir(anchor_path):
                img_path = os.path.join(anchor_path, anchor_img_name)
                anchor_img = load_and_preprocess(img_path)
                if anchor_img is None:
                    continue

                for positive_img_name in os.listdir(positive_path):
                    positive_img = load_and_preprocess(os.path.join(positive_path, positive_img_name))
                    if positive_img is None:
                        continue
                    # Get a negative sample
                    neg_individual = np.random.choice([ind for j, ind in enumerate(individuals) if j != i])
                    neg_hand = np.random.choice(['L', 'R'])
                    negative_path = os.path.join(root_folder, neg_individual, neg_hand)
                    negative_img_name = np.random.choice(os.listdir(negative_path))
                    negative_img = load_and_preprocess(os.path.join(negative_path, negative_img_name))
                    if negative_img is None:
                        continue

                    # Append to lists
                    anchors.append(anchor_img)
                    positives.append(positive_img)
                    negatives.append(negative_img)

    # Convert to numpy arrays for training
    anchors = np.array(anchors).reshape(-1, 192, 192, 1)
    positives = np.array(positives).reshape(-1, 192, 192, 1)
    negatives = np.array(negatives).reshape(-1, 192, 192, 1)

    input_anchor = Input(shape=(192, 192, 1))
    input_positive = Input(shape=(192, 192, 1))
    input_negative = Input(shape=(192, 192, 1))

    embedding_model = model2.create_unet_model((192, 192))

    embedding_anchor = embedding_model(input_anchor)
    embedding_positive = embedding_model(input_positive)
    embedding_negative = embedding_model(input_negative)

    output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

    triplet_model = Model([input_anchor, input_positive, input_negative], output)
    triplet_model.compile(optimizer='adam', loss=triplet_loss)

    y_dummy = np.empty((anchors.shape[0], 3 * 128))
    triplet_model.fit([anchors, positives, negatives], y_dummy, epochs=12, batch_size=16)

    saved_model_path = os.path.join(".", 'saved_model2')
    save_model(triplet_model, saved_model_path)
