import pytest

import losses
import numpy as np

@pytest.mark.parametrize(
    "B,M",
    [
        (100, 1),
        (100, 3),
        (100, 6),
        (100, 12),
        (200, 1),
        (200, 3),
        (200, 6),
        (200, 12),
    ])
def test_class_loss(B, M):
    np.random.seed(71)

    logits = np.random.uniform(size=(M, B, 10))
    y = np.random.choice(range(10), B)

    class_loss_tf1 = losses.compute_class_loss_tf1(logits, y).numpy()
    class_loss_tf2 = losses.compute_class_loss_tf2(logits, y).numpy()

    assert np.isclose(class_loss_tf1, class_loss_tf2), \
        "Class Loss's TF1 & TF2 return the same result"