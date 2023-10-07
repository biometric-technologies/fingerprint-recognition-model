import model
import os
import numpy as np
import cv2
from keras.utils import to_categorical
from keras.models import save_model


def resize_and_pad(img, target_size=192):
    # Calculate scale factors
    height_scale = target_size / float(img.shape[0])
    width_scale = target_size / float(img.shape[1])

    # Resize the image while maintaining aspect ratio
    if img.shape[0] > img.shape[1]:
        new_height = target_size
        new_width = int(img.shape[1] * width_scale)
    else:
        new_width = target_size
        new_height = int(img.shape[0] * height_scale)

    resized_img = cv2.resize(img, (new_width, new_height))

    # Pad the image with zeros to make it square (192x192)
    top_pad = (target_size - new_height) // 2
    bottom_pad = target_size - new_height - top_pad
    left_pad = (target_size - new_width) // 2
    right_pad = target_size - new_width - left_pad
    padded_img = cv2.copyMakeBorder(resized_img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT,
                                    value=[0, 0, 0])

    return padded_img, width_scale, height_scale


def resize_label_map(label_map, target_size=192):
    # Calculate scale factors
    height_scale = target_size / float(label_map.shape[0])
    width_scale = target_size / float(label_map.shape[1])

    # Resize while maintaining aspect ratio
    if label_map.shape[0] > label_map.shape[1]:
        new_height = target_size
        new_width = int(label_map.shape[1] * width_scale)
    else:
        new_width = target_size
        new_height = int(label_map.shape[0] * height_scale)

    resized_label_map = cv2.resize(label_map, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Pad the label map to make it square
    top_pad = (target_size - new_height) // 2
    bottom_pad = target_size - new_height - top_pad
    left_pad = (target_size - new_width) // 2
    right_pad = target_size - new_width - left_pad
    padded_label_map = cv2.copyMakeBorder(resized_label_map, top_pad, bottom_pad, left_pad, right_pad,
                                          cv2.BORDER_CONSTANT, value=0)

    return padded_label_map


if __name__ == '__main__':
    model = model.create_unet_model((192, 192, 1))
    folder = "./fingerprint-v5-master/000/L"
    all_label_maps = []
    all_angles = []
    all_images = []
    for dirpath, _, filenames in os.walk(folder):
        for file in filenames:
            full_file_path = os.path.join(dirpath, file)
            base_name, ext = os.path.splitext(file)
            if ext.lower() not in [".bmp"]:
                continue
            img = cv2.imread(full_file_path, cv2.IMREAD_GRAYSCALE)
            meta = np.load(full_file_path.replace(".bmp", ".npz"))
            label_map = meta["minutiaes"]
            angles = meta["angles"]
            resized_img, _, _ = resize_and_pad(img)
            resized_label_map = resize_label_map(label_map)
            resized_angles = resize_label_map(angles)

            all_images.append(resized_img)
            all_label_maps.append(resized_label_map)
            all_angles.append(resized_angles)

    stacked_label_maps = np.stack(all_label_maps, axis=0)
    # train model
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam',
                  loss={'classification': 'categorical_crossentropy', 'orientation': 'mean_squared_error'},
                  metrics={'classification': 'accuracy'})
    x_train = np.array(all_images)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    y_train_label = to_categorical(stacked_label_maps, num_classes=3)
    y_train_angles = np.stack(all_angles, axis=0)
    print(f"X Images shape: {x_train.shape}")
    print(f"Y Labels shape: {y_train_label.shape}")
    print(f"Y Orientations shape: {y_train_angles.shape}")
    model.fit(x_train, [y_train_label, y_train_angles], batch_size=4, epochs=12, validation_split=0.2)
    saved_model_path = os.path.join(".", 'saved_model')
    save_model(model, saved_model_path)
