import numpy as np

import model


def test_embeddings_are_unit_vectors_that_use_every_direction():
    embedding_model = model.create_embedding_model((192, 192, 1))
    assert embedding_model.output_shape == (None, 128)

    rng = np.random.default_rng(0)
    images = rng.random((4, 192, 192, 1)).astype(np.float32)
    embeddings = embedding_model.predict(images, verbose=0)

    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
    # The template layer is linear. With a ReLU before the normalization every
    # component would be >= 0, and the embeddings would fill only one orthant.
    assert (embeddings < 0).any()
