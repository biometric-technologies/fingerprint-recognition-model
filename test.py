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


def evaluate_model(model, test_images, test_labels, threshold=1.0):
    """Verification metrics (ISO/IEC 19795-1) over all pairs of test images.

    A pair is accepted as the same finger when the distance is below the threshold.
    FMR: share of impostor pairs accepted; FNMR: share of genuine pairs rejected.
    """
    embeddings = model.predict(test_images)
    genuine_distances = []
    impostor_distances = []
    for i in range(len(embeddings)):
        d = np.linalg.norm(embeddings[i + 1:] - embeddings[i], axis=1)
        same = test_labels[i + 1:] == test_labels[i]
        genuine_distances.extend(d[same].tolist())
        impostor_distances.extend(d[~same].tolist())
    g = np.array(genuine_distances)
    im = np.array(impostor_distances)
    fmr = float(np.mean(im < threshold))
    fnmr = float(np.mean(g >= threshold))
    # EER: threshold where FMR and FNMR are closest
    candidates = np.unique(np.concatenate([g, im]))
    errors = [(max(np.mean(im < t), np.mean(g >= t)), t) for t in candidates]
    eer, eer_threshold = min(errors)
    # FNMR at fixed FMR
    fnmr_at = {}
    sorted_im = np.sort(im)
    for target in (1e-2, 1e-3, 1e-4):
        k = int(target * len(sorted_im))
        if k < 1:
            continue
        t = sorted_im[k - 1] if k <= len(sorted_im) else np.inf
        fnmr_at[target] = (float(np.mean(g >= t)), float(t))
    return fmr, fnmr, float(eer), float(eer_threshold), fnmr_at, genuine_distances, impostor_distances


if __name__ == '__main__':
    saved_model_path = "saved_model"
    loaded_model = load_model(saved_model_path, custom_objects={"triplet_loss": triplet_loss})
    test_images, test_labels = load_dataset("fingerprint-v5-master-test")
    test_labels = preprocess_labels(test_labels)
    embedding_model = loaded_model.layers[3]
    fmr, fnmr, eer, eer_threshold, fnmr_at, genuine_distances, impostor_distances = \
        evaluate_model(embedding_model, test_images, test_labels)
    print(f"Genuine pairs: {len(genuine_distances)}, impostor pairs: {len(impostor_distances)}")
    print(f"At threshold 1.0: FMR {fmr * 100:.3f}%, FNMR {fnmr * 100:.2f}%")
    print(f"EER: {eer * 100:.2f}% at distance {eer_threshold:.3f}")
    for target, (value, t) in fnmr_at.items():
        print(f"FNMR at FMR {target:g}: {value * 100:.2f}% (distance {t:.3f})")
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
