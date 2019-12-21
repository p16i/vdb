import tensorflow as tf
import tensorflow_probability as tfp

import losses

from nets.base import BaseNet

class Net(BaseNet):
    """
    ResNet20 implementation based on https://github.com/akamaster/pytorch_resnet_cifar10
    """

    def __init__(self, architecture, in_shape, num_class, cov_type,  batch_norm=True, beta=1e-3, M=1):
        super(Net, self).__init__(architecture, cov_type, num_class, beta, M)

        self.encoder = tf.keras.Sequential(
            [
                ResNet(BasicBlock, [3, 3, 3],
                num_classes=self.parameters_for_latent,
                batch_norm=batch_norm),
            ]
        )

class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10, batch_norm=True):
        super(ResNet, self).__init__()
        self.batch_norm = batch_norm
        self.in_planes = 16

        self.pad1 = tf.keras.layers.ZeroPadding2D((1, 1))
        self.conv1 = tf.keras.layers.Conv2D(self.in_planes, kernel_size=3, strides=1, padding="SAME", use_bias=False, kernel_initializer="he_normal")
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, batch_norm=batch_norm)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, batch_norm=batch_norm)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, batch_norm=batch_norm)
        self.fcout =  tf.keras.layers.Dense(num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, batch_norm):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride, option='A', batch_norm=batch_norm))
            self.in_planes = planes * block.expansion

        return tf.keras.Sequential(layers)


    def call(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)

        out = tf.nn.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = tf.nn.avg_pool2d(out, out.shape[1], strides=1, padding="VALID")
      
        out = tf.reshape(out, (tf.shape(out)[0],  -1))
        
        out = self.fcout(out)
        return out

class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', batch_norm=True):
        super(BasicBlock, self).__init__()
        self.batch_norm = batch_norm

        self.stride = stride
        
        self.conv1 = tf.keras.layers.Conv2D(planes, kernel_size=3, strides=stride, padding="SAME", use_bias=False, kernel_initializer="he_normal")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.padding = tf.keras.layers.ZeroPadding2D(1)
        self.conv2 = tf.keras.layers.Conv2D(planes, kernel_size=3, strides=1, padding="VALID", use_bias=False, kernel_initializer="he_normal")
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.shortcut = tf.keras.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = lambda x: tf.pad(x[:, ::2, ::2, :], [[0, 0], [0, 0], [0, 0], [planes // 4, planes // 4]], "CONSTANT", constant_values=0)
            else:
                raise SystemError("Not implemented")

    def call(self, x):
        out = self.conv1(x)

        if self.batch_norm:
            out = self.bn1(out)

        out = tf.nn.relu(out)
        out_padded = self.padding(out)

        out = self.conv2(out_padded)

        if self.batch_norm:
          out = self.bn2(out)

        sc = self.shortcut(x)

        out = out + sc
        out = tf.nn.relu(out)
        return out