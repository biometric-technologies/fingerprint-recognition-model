from math import ceil
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
import os
import numpy as np
import cv2
from random import choice, sample


class CasiaV5TripletGenerator(Sequence):
    def __init__(self, root_folder, subjects, batch_size):
        self.root_folder = root_folder
        self.batch_size = batch_size
        self.subjects = subjects
        self.total_samples = sum(
            [len(os.listdir(os.path.join(root_folder, subj, 'L'))) for subj in subjects])
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (192, 192))
        img = img / 255.0
        return img

    def _get_triplet(self, subjects, subj, hand):
        anchor_finger_idx = np.random.choice(range(4))  # Assuming 4 fingers
        anchor_sample_idx = np.random.choice(range(5))  # Assuming at least 5 samples per finger

        anchor_path = os.path.join(self.root_folder, subj, hand,
                                   f"{subj}_{hand}{anchor_finger_idx}_{anchor_sample_idx}.bmp")
        anchor_img = self._load_image(anchor_path)

        # Select a different sample index for the positive sample
        positive_sample_idx = choice([i for i in range(5) if i != anchor_sample_idx])
        positive_path = os.path.join(self.root_folder, subj, hand,
                                     f"{subj}_L{anchor_finger_idx}_{positive_sample_idx}.bmp")
        positive_img = self._load_image(positive_path)

        # Get a negative sample (different subject)
        negative_subj = choice([s for s in subjects if s != subj])
        negative_finger_idx = np.random.choice(range(4))
        negative_sample_idx = np.random.choice(range(5))
        negative_path = os.path.join(self.root_folder, negative_subj, hand,
                                     f"{negative_subj}_{hand}{negative_finger_idx}_{negative_sample_idx}.bmp")
        negative_img = self._load_image(negative_path)

        return anchor_img, positive_img, negative_img

    def _get_triplets(self, subjects):
        anchors, positives, negatives = [], [], []
        for _ in range(self.batch_size):
            subj = choice(subjects)
            for hand in ["L", "R"]:
                anchor, positive, negative = self._get_triplet(subjects, subj, hand)
                anchors.append(self.datagen.random_transform(anchor))
                positives.append(self.datagen.random_transform(positive))
                negatives.append(self.datagen.random_transform(negative))

        return [np.array(anchors), np.array(positives), np.array(negatives)], np.zeros(self.batch_size)

    def __len__(self):
        return self.total_samples // self.batch_size

    def __getitem__(self, idx):
        return self._get_triplets(self.subjects)


def create_casia_v5_generators(root_folder, batch_size, split_ratio=0.8):
    all_subjects = os.listdir(root_folder)
    np.random.shuffle(all_subjects)
    split_at = int(len(all_subjects) * split_ratio)
    train_subjects = all_subjects[:split_at]
    validation_subjects = all_subjects[split_at:]
    train_gen = CasiaV5TripletGenerator(root_folder, train_subjects, batch_size)
    val_gen = CasiaV5TripletGenerator(root_folder, validation_subjects, batch_size)
    return train_gen, val_gen
