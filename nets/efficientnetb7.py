import tensorflow as tf

from nets.base import BaseNet

from efficientnet.tfkeras import EfficientNetB7

class Net(BaseNet):
    """
    Please see resnet20.py for more information about the implementation
    """

    def __init__(self, architecture, in_shape, num_class, cov_type,  batch_norm=True, beta=1e-3, M=1):
        super(Net, self).__init__(architecture, cov_type, num_class, beta, M)

        self.encoder = EfficientNetB7(
            weights=None,
            classes=self.parameters_for_latent
        )
