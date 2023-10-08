import os
import cv2
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, concatenate
from keras.models import Model, save_model
import model
from tripplet_generator import create_casia_v5_generators
import matplotlib.pyplot as plt


def triplet_loss(y_true, y_pred, margin=1.0):
    anchor, positive, negative = y_pred[:, :128], y_pred[:, 128:256], y_pred[:, 256:]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
    return tf.reduce_mean(loss)


if __name__ == '__main__':
    root_folder = 'fingerprint-v5-master'

    embedding_model = model.create_embedding_model((192, 192, 1))
    print(embedding_model.summary())

    input_anchor = Input(shape=(192, 192, 1))
    input_positive = Input(shape=(192, 192, 1))
    input_negative = Input(shape=(192, 192, 1))

    embedding_anchor = embedding_model(input_anchor)
    embedding_positive = embedding_model(input_positive)
    embedding_negative = embedding_model(input_negative)

    output = concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)
    triplet_model = Model([input_anchor, input_positive, input_negative], output)
    triplet_model.compile(optimizer='adam', loss=triplet_loss)

    callbacks_list = [
        ModelCheckpoint(
            filepath='fingerprint_model/checkpoint_{epoch:02d}.hdf5',
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            mode='auto',
            save_freq='epoch',
            options=None,
            initial_value_threshold=None
        )
    ]

    train_gen, val_gen = create_casia_v5_generators(root_folder, 64)

    history = triplet_model.fit(x=train_gen, validation_data=val_gen, callbacks=callbacks_list, epochs=256)

    saved_model_path = os.path.join(".", 'saved_model')
    save_model(triplet_model, saved_model_path)

    # Plotting training loss
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('train_metrics.png', dpi=300, bbox_inches='tight')
