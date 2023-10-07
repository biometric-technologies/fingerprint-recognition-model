from math import ceil

from keras.utils import Sequence
import os
import numpy as np
import cv2
from random import choice, sample


class CasiaV5TripletGenerator(Sequence):
    def __init__(self, root_folder, batch_size):
        self.root_folder = root_folder
        self.batch_size = batch_size
        self.subjects = os.listdir(root_folder)
        self.total_samples = sum([len(os.listdir(os.path.join(root_folder, subj, 'L'))) for subj in self.subjects])

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (192, 192))
        img = img / 255.0
        return img

    def _get_triplet(self, subj):
        # Get anchor and positive samples
        hand = 'L'
        anchor_finger_idx = np.random.choice(range(4))  # Assuming 5 fingers
        anchor_sample_idx = np.random.choice(range(5))  # Assuming at least 5 samples per finger

        anchor_path = os.path.join(self.root_folder, subj, hand, f"{subj}_L{anchor_finger_idx}_{anchor_sample_idx}.bmp")
        anchor_img = self._load_image(anchor_path)

        # Select a different sample index for the positive sample
        positive_sample_idx = choice([i for i in range(5) if i != anchor_sample_idx])
        positive_path = os.path.join(self.root_folder, subj, hand,
                                     f"{subj}_L{anchor_finger_idx}_{positive_sample_idx}.bmp")
        positive_img = self._load_image(positive_path)

        # Get a negative sample (different subject)
        negative_subj = choice([s for s in self.subjects if s != subj])
        negative_finger_idx = np.random.choice(range(4))
        negative_sample_idx = np.random.choice(range(5))
        negative_path = os.path.join(self.root_folder, negative_subj, hand,
                                     f"{negative_subj}_L{negative_finger_idx}_{negative_sample_idx}.bmp")
        negative_img = self._load_image(negative_path)

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return self.total_samples // self.batch_size

    def __getitem__(self, idx):
        anchors, positives, negatives = [], [], []
        for _ in range(self.batch_size):
            subj = choice(self.subjects)
            anchor, positive, negative = self._get_triplet(subj)
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        return [np.array(anchors), np.array(positives), np.array(negatives)], np.zeros(self.batch_size)
