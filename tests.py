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

    num_classes = 10

    logits = np.random.uniform(size=(M, B, 10))
    y = np.random.choice(range(num_classes), B)

    class_loss_tf1 = losses.compute_vdb_class_loss_tf1(logits, y, num_classes)\
        .numpy()
    class_loss_tf2 = losses.compute_vdb_class_loss(logits, y, num_classes)\
        .numpy()

    assert np.isclose(class_loss_tf1, class_loss_tf2), \
        "Class Loss's TF1 & TF2 return the same result"