import cv2
import os
import csv
import fingerprint_enhancer
import fingerprint_feature_extractor
import numpy
import math
import numpy as np

image_extensions = ['.bmp']

if __name__ == '__main__':
    folder = "./fingerprint-v5-master/000/L"
    for dirpath, _, filenames in os.walk(folder):
        print(f"iterating {len(filenames)} files in dir {dirpath}")
        for file in filenames:
            full_file_path = os.path.join(dirpath, file)
            base_name, ext = os.path.splitext(file)
            if ext.lower() not in image_extensions:
                continue
            img = cv2.imread(full_file_path, cv2.IMREAD_GRAYSCALE)
            ench = fingerprint_enhancer.enhance_Fingerprint(img)
            terminations, bifurcations = fingerprint_feature_extractor.extract_minutiae_features(ench)
            label_map = np.zeros(img.shape, dtype=np.int8)
            angles_map = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.int8)
            for term in terminations:
                label_map[term.locX, term.locY] = 1
                angles_map[term.locX, term.locY, 0] = term.Orientation[0]
            for bif in bifurcations:
                label_map[bif.locX, bif.locY] = 2
                angles_map[bif.locX, bif.locY, :] = bif.Orientation
            res_path = os.path.join(dirpath, base_name + ".npz")
            os.remove(res_path)
            numpy.savez(res_path, minutiaes=label_map, angles=angles_map)
