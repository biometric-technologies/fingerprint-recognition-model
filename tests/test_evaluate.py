import numpy as np
import pytest

from conftest import load_script

evaluation = load_script("fingerprint_evaluation", "test.py")


class FixedEmbeddings:
    """Stands in for the embedding model: returns the given embeddings."""

    def __init__(self, embeddings):
        self.embeddings = np.asarray(embeddings, dtype=float)

    def predict(self, images):
        assert len(images) == len(self.embeddings)
        return self.embeddings


def test_metrics_count_genuine_and_impostor_pairs():
    # Fingers 0 and 1, two samples each, on the unit circle:
    # genuine pairs at distance 0.2 (finger 0) and 1.2 (finger 1),
    # impostor pairs at about 1.41 and one close impostor at 0.5.
    angle = np.array([0.0, 0.2, np.pi / 2, np.pi / 2 + 1.2635])
    embeddings = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    labels = np.array([0, 0, 1, 1])
    distances = np.linalg.norm(embeddings[:, None] - embeddings[None], axis=2)
    genuine = [distances[0, 1], distances[2, 3]]
    impostor = [distances[i, j] for i in range(4) for j in range(i + 1, 4) if labels[i] != labels[j]]

    fmr, fnmr, eer, eer_threshold, fnmr_at, g, im = evaluation.evaluate_model(
        FixedEmbeddings(embeddings), np.zeros((4, 1)), labels, threshold=1.0)

    np.testing.assert_allclose(sorted(g), sorted(genuine))
    np.testing.assert_allclose(sorted(im), sorted(impostor))
    assert len(g) == 2 and len(im) == 4
    # at 1.0: impostors closer than 1.0 are accepted, genuine pairs at 1.0 or more rejected
    assert fmr == pytest.approx(np.mean(np.array(impostor) < 1.0))
    assert fnmr == pytest.approx(np.mean(np.array(genuine) >= 1.0))
    # the EER is the smallest worst-case error over the candidate thresholds
    best = min(max(np.mean(np.array(impostor) < t), np.mean(np.array(genuine) >= t))
               for t in genuine + impostor)
    assert eer == pytest.approx(best)
    # too few impostor pairs for FNMR at FMR 1% or lower
    assert fnmr_at == {}


def test_perfect_separation_gives_zero_error():
    embeddings = [[1, 0], [1, 0], [0, 1], [0, 1]]
    fmr, fnmr, eer, _, _, _, _ = evaluation.evaluate_model(
        FixedEmbeddings(embeddings), np.zeros((4, 1)), np.array([0, 0, 1, 1]))
    assert (fmr, fnmr, eer) == (0.0, 0.0, 0.0)
