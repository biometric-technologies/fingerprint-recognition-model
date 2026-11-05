import os

import cv2
import numpy as np
import pytest

import tripplet_generator
from tripplet_generator import CasiaV5TripletGenerator, create_casia_v5_generators

SUBJECTS = ["000", "001", "002", "003", "004"]


@pytest.fixture
def casia(tmp_path):
    """A small tree in the CASIA-FingerprintV5 layout: <subject>/<L|R>/<subject>_<hand><finger>_<sample>.bmp."""
    rng = np.random.default_rng(1)
    for subject in SUBJECTS:
        for hand in "LR":
            folder = tmp_path / subject / hand
            folder.mkdir(parents=True)
            for finger in range(4):
                for sample in range(5):
                    image = rng.integers(0, 256, (64, 48), dtype=np.uint8)
                    cv2.imwrite(str(folder / f"{subject}_{hand}{finger}_{sample}.bmp"), image)
    return str(tmp_path)


def identity(path):
    """(subject, hand, finger, sample) from a CASIA file name."""
    name = os.path.splitext(os.path.basename(path))[0]
    subject, rest = name.split("_", 1)
    hand_finger, sample = rest.split("_")
    return subject, hand_finger[0], int(hand_finger[1]), int(sample)


def test_batches_hold_two_triplets_per_drawn_subject(casia):
    generator = CasiaV5TripletGenerator(casia, SUBJECTS, batch_size=3)
    (anchors, positives, negatives), labels = generator[0]
    for images in (anchors, positives, negatives):
        assert images.shape == (6, 192, 192, 1)
        assert 0.0 <= images.min() and images.max() <= 1.0
    assert labels.shape == (6,)
    # 5 subjects x 20 left-hand images x 2 hands
    assert generator.total_samples == 200
    assert len(generator) == 200 // 3


def test_triplets_pair_the_same_finger_and_contrast_another(casia, monkeypatch):
    loaded = []
    monkeypatch.setattr(CasiaV5TripletGenerator, "_load_image", lambda self, path: loaded.append(path))
    generator = CasiaV5TripletGenerator(casia, SUBJECTS, batch_size=1)
    for _ in range(200):
        loaded.clear()
        generator._get_triplet(SUBJECTS, "002", "R")
        anchor, positive, negative = (identity(p) for p in loaded)
        for path in loaded:
            assert os.path.isfile(path), path
        assert anchor[:3] == ("002", "R", anchor[2]) and positive[:3] == anchor[:3]
        assert positive[3] != anchor[3], "positive is the anchor image itself"
        assert negative[:3] != anchor[:3], "negative is the anchor's own finger"


def test_train_and_validation_subjects_are_disjoint_and_reproducible(casia):
    train, validation = create_casia_v5_generators(casia, batch_size=2)
    assert set(train.subjects).isdisjoint(validation.subjects)
    assert sorted(train.subjects + validation.subjects) == SUBJECTS
    assert len(train.subjects) == 4
    again, _ = create_casia_v5_generators(casia, batch_size=2)
    assert again.subjects == train.subjects


def test_augmentation_does_not_mirror(casia):
    # A mirrored fingerprint is not a real finger
    datagen = CasiaV5TripletGenerator(casia, SUBJECTS, batch_size=1).datagen
    assert not datagen.horizontal_flip and not datagen.vertical_flip
