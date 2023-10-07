from keras.utils import Sequence
import os
import numpy as np
import cv2


class TripletGenerator(Sequence):
    def __init__(self, root_folder, batch_size):
        self.root_folder = root_folder
        self.individuals = os.listdir(root_folder)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.individuals) * 2 / self.batch_size))  # 2 for both hands

    def load_and_preprocess(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (192, 192))
        img = img / 255.0
        return img

    def __getitem__(self, index):
        anchors = []
        positives = []
        negatives = []

        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.individuals) * 2)

        for i in range(start_idx, end_idx):
            ind_idx = i // 2
            current_individual = self.individuals[ind_idx]
            hand = 'L' if i % 2 == 0 else 'R'

            # Paths
            anchor_path = os.path.join(self.root_folder, current_individual, hand)
            positive_hand = 'R' if hand == 'L' else 'L'
            positive_path = os.path.join(self.root_folder, current_individual, positive_hand)

            # Randomly select images
            anchor_img_name = np.random.choice(os.listdir(anchor_path))
            positive_img_name = np.random.choice(os.listdir(positive_path))
            neg_individual = np.random.choice([ind for j, ind in enumerate(self.individuals) if j != ind_idx])
            neg_hand = np.random.choice(['L', 'R'])
            negative_img_name = np.random.choice(os.listdir(os.path.join(self.root_folder, neg_individual, neg_hand)))

            # Load images
            anchor_img = self.load_and_preprocess(os.path.join(anchor_path, anchor_img_name))
            positive_img = self.load_and_preprocess(os.path.join(positive_path, positive_img_name))
            negative_img = self.load_and_preprocess(
                os.path.join(self.root_folder, neg_individual, neg_hand, negative_img_name))

            if anchor_img is None or positive_img is None or negative_img is None:
                continue

            anchors.append(anchor_img)
            positives.append(positive_img)
            negatives.append(negative_img)

        anchors = np.array(anchors).reshape(-1, 192, 192, 1)
        positives = np.array(positives).reshape(-1, 192, 192, 1)
        negatives = np.array(negatives).reshape(-1, 192, 192, 1)
        y_dummy = np.empty((anchors.shape[0], 3 * 128))
        return [anchors, positives, negatives], y_dummy
