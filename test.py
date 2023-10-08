import tensorflow as tf
from keras.models import Model, load_model
import cv2
import os
import numpy as np
from train import triplet_loss


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (192, 192))
    img = cv2.equalizeHist(img)
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
    for n in os.listdir(root_folder):
        for lr in os.listdir(os.path.join(root_folder, n)):
            for img_file in os.listdir(os.path.join(root_folder, n, lr)):
                if img_file.endswith('.bmp'):
                    # Load the image
                    img_path = os.path.join(root_folder, n, lr, img_file)
                    img = preprocess_image(img_path)
                    images.append(img)
                    # Use the directory structure as a label (optional)
                    label = f"{n}_{lr}"
                    labels.append(label)
    return np.array(images), np.array(labels)


def evaluate_model(model, test_images, test_labels):
    threshold = 0.5
    num_correct = 0
    embeddings = model.predict(test_images)
    pairwise_distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            if pairwise_distances[i][j] < threshold and test_labels[i] == test_labels[j]:
                num_correct += 1
            elif pairwise_distances[i][j] >= threshold and test_labels[i] != test_labels[j]:
                num_correct += 1

    return num_correct / (len(embeddings) * len(embeddings))


if __name__ == '__main__':
    saved_model_path = "saved_model"
    loaded_model = load_model(saved_model_path, custom_objects={"triplet_loss": triplet_loss})
    test_images, test_labels = load_dataset("fingerprint-v5-master-test")
    test_labels = preprocess_labels(test_labels)
    embedding_model = loaded_model.layers[3]
    accuracy = evaluate_model(embedding_model, test_images, test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
