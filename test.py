import tensorflow as tf
from keras.models import Model, load_model
import cv2
import os
import numpy as np
from train import triplet_loss
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (192, 192))
    img = cv2.equalizeHist(img)
    img = cv2.bitwise_not(img)
    img = img / 255.0
    return img


def preprocess_labels(labels):
    # Create a mapping from unique label strings to integers
    unique_labels = np.unique(labels)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}

    # Convert the array of string labels to integer IDs
    int_labels = np.array([label_to_id[label] for label in labels])

    return int_labels


def load_dataset(root_folder):
    images = []
    labels = []

    # Iterate through the directories
    for subj in os.listdir(root_folder):
        for hand in os.listdir(os.path.join(root_folder, subj)):
            for finger_id in range(4):
                for sample_id in range(5):
                    label = f"{subj}_{hand}{finger_id}"
                    img_file = f"{subj}_{hand}{finger_id}_{sample_id}.bmp"
                    img_path = os.path.join(root_folder, subj, hand, img_file)
                    img = preprocess_image(img_path)
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)


def evaluate_model(model, test_images, test_labels):
    threshold = 1.0
    tp, tn, fp, fn = 0, 0, 0, 0
    genuine_distances = []
    impostor_distances = []

    embeddings = model.predict(test_images)
    pairwise_distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            if i == j:
                continue
            if test_labels[i] == test_labels[j]:
                genuine_distances.append(pairwise_distances[i][j])
                if pairwise_distances[i][j] >= threshold:
                    tp += 1
                elif pairwise_distances[i][j] < threshold:
                    fp += 1
            if test_labels[i] != test_labels[j]:
                impostor_distances.append(pairwise_distances[i][j])
                if pairwise_distances[i][j] < threshold:
                    tn += 1
                elif pairwise_distances[i][j] >= threshold:
                    fn += 1
    total = tp + tn + fp + fn
    FAR = fp / max(total, 1) * 100
    FRR = fn / max(total, 1) * 100
    accuracy = (tp + tn) / total * 100
    APCER = fp / (tn + fp)
    BPCER = fn / (tp + fn)
    ACER = (APCER + BPCER) / 2 * 100

    return accuracy, FAR, FRR, ACER, genuine_distances, impostor_distances


if __name__ == '__main__':
    saved_model_path = "saved_model"
    loaded_model = load_model(saved_model_path, custom_objects={"triplet_loss": triplet_loss})
    test_images, test_labels = load_dataset("fingerprint-v5-master-test")
    test_labels = preprocess_labels(test_labels)
    embedding_model = loaded_model.layers[3]
    accuracy, FAR, FRR, ACER, genuine_distances, impostor_distances = evaluate_model(embedding_model, test_images,
                                                                                     test_labels)
    print(f"False Accept Rate %: {FAR:.2f}")
    print(f"False Rejection Rate %: {FRR:.2f}")
    print(f"Accuracy %: {accuracy:.2f}")
    print(f"ACER %: {ACER:.2f}")
    plt.figure(figsize=(10, 5))
    # Plotting the genuine distances
    plt.hist(genuine_distances, bins=50, alpha=0.5, color='blue', label='Genuine pairs')
    # Plotting the impostor distances
    plt.hist(impostor_distances, bins=50, alpha=0.5, color='red', label='Impostor pairs')
    plt.title('Distribution of pairwise distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()

    plt.savefig('test_metrics.png', dpi=300, bbox_inches='tight')
