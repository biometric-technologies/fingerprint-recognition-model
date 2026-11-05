import numpy as np
import tensorflow as tf

from train import triplet_loss


def batch(anchor, positive, negative):
    """One row per triplet: anchor, positive and negative 128-value embeddings side by side."""
    return tf.constant(np.concatenate([anchor, positive, negative], axis=1), dtype=tf.float32)


def unit(index, n=1):
    v = np.zeros((n, 128))
    v[:, index] = 1.0
    return v


def test_no_loss_when_the_negative_is_farther_by_more_than_the_margin():
    # positive at distance 0, negative at squared distance 2
    y_pred = batch(unit(0), unit(0), unit(1))
    assert float(triplet_loss(None, y_pred)) == 0.0


def test_loss_is_the_margin_when_positive_and_negative_are_equally_far():
    y_pred = batch(unit(0), unit(1), unit(2))
    np.testing.assert_allclose(float(triplet_loss(None, y_pred)), 0.3, rtol=1e-6)
    np.testing.assert_allclose(float(triplet_loss(None, y_pred, margin=0.5)), 0.5, rtol=1e-6)


def test_loss_is_the_mean_over_the_batch():
    # first triplet: loss 0; second: squared distances 2 and 0, loss 2 + 0.3
    anchor = np.vstack([unit(0), unit(0)])
    positive = np.vstack([unit(0), unit(1)])
    negative = np.vstack([unit(1), unit(0)])
    np.testing.assert_allclose(float(triplet_loss(None, batch(anchor, positive, negative))), 2.3 / 2, rtol=1e-6)
