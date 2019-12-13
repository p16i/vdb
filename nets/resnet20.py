import tensorflow as tf
import tensorflow_probability as tfp

import losses

from nets.base import BaseNet

def debug(*args):
    pass

class Net(BaseNet):
    """
    ResNet20 implementation based on https://github.com/akamaster/pytorch_resnet_cifar10
    """

    def __init__(self, architecture, in_shape=None,  batch_norm=True, beta=1e-3, M=1):
        super(Net, self).__init__()
        # assert tuple(in_shape) == (3, 32, 32)

        latent_dim = architecture["z"]
        self.latent_dim = latent_dim

        num_cov_entries = latent_dim
        
        out_dim = 10
        # self.input_grad = input_grad
        self.out_dim = out_dim
        self.encoder = tf.keras.Sequential(
            [
                ResNet(BasicBlock, [3, 3, 3], num_classes=out_dim, batch_norm=batch_norm),
                tf.keras.layers.Dense(latent_dim + num_cov_entries),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=out_dim),
            ]
        )

        self.prior = tfp.distributions.Normal(0, 1)

        self.compute_info_loss = losses.compute_info_loss_diag_cov

        self.beta = beta
        self.M = M


    def encode(self, x):
        entries = self.encoder(x)
        mean = entries[:, :self.latent_dim]
        cov_entries = entries[:, self.latent_dim:]

        cov_entries  = tf.nn.softplus(cov_entries - 5.)

        return tfp.distributions.Normal(mean, cov_entries)

class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10, batch_norm=True):
        super(ResNet, self).__init__()
        self.batch_norm = batch_norm
        self.in_planes = 16

        self.pad1 = tf.keras.layers.ZeroPadding2D((1, 1))
        self.conv1 = tf.keras.layers.Conv2D(self.in_planes, kernel_size=3, strides=1, padding="SAME", use_bias=False, kernel_initializer="he_normal")
        self.bn1 = tf.keras.layers.BatchNormalization()

        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, batch_norm=batch_norm)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, batch_norm=batch_norm)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, batch_norm=batch_norm)
        self.fcout =  tf.keras.layers.Dense(10)

        # self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, batch_norm):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride, option='A', batch_norm=batch_norm))
            self.in_planes = planes * block.expansion

        return tf.keras.Sequential(layers)


    def call(self, x):
        debug("input", x)
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = tf.nn.relu(out)
        # print("layer 1")
        out = self.layer1(out)
        # print("layer 2")
        out = self.layer2(out)
        # print("layer 3")
        out = self.layer3(out)
        
        debug("out layer3", out)

        out = tf.nn.avg_pool2d(out, out.shape[1], strides=1, padding="VALID")
      
        debug("avg_pool2d", out)
        out = tf.reshape(out, (tf.shape(out)[0],  -1))
        
        out = self.fcout(out)
        debug("dense", out)
        return out

# def _weights_init(m):
    # todo: finding He initialization in TF2
    # if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    #     nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    # return m

class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', batch_norm=True):
        super(BasicBlock, self).__init__()
        self.batch_norm = batch_norm

        self.stride = stride
        
        # self.padding = lambda x: x
        self.conv1 = tf.keras.layers.Conv2D(planes, kernel_size=3, strides=stride, padding="SAME", use_bias=False, kernel_initializer="he_normal")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.padding = tf.keras.layers.ZeroPadding2D(1)
        self.conv2 = tf.keras.layers.Conv2D(planes, kernel_size=3, strides=1, padding="VALID", use_bias=False, kernel_initializer="he_normal")
        self.bn2 = tf.keras.layers.BatchNormalization()

        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = tf.keras.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = lambda x: tf.pad(x[:, ::2, ::2, :], [[0, 0], [0, 0], [0, 0], [planes // 4, planes // 4]], "CONSTANT", constant_values=0)
                # self.shortcut = LambdaLayer(lambda x:
                #                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                #                                   0))
                # self.shortcut.add
            # elif option == 'B':
            #     self.shortcut = nn.Sequential(
            #         nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            #         nn.BatchNorm2d(self.expansion * planes)
            #     )

    def call(self, x):
        # print(f"=========== stride {self.stride} =========")
        # print("x", tf.shape(x), tf.shape(x[:, ::2, ::2, :]))
        debug("in", x)

        
        out = self.conv1(x)
        
        debug("out-1", out)
        
        if self.batch_norm:
            out = self.bn1(out)

        out = tf.nn.relu(out)
        out_padded = self.padding(out)
        debug("out-1-padded", out_padded)
        out = self.conv2(out_padded)
        debug("out-2", out)

        if self.batch_norm:
          out = self.bn2(out)
        sc = self.shortcut(x)
        debug("shortcut", sc)
        out = out + sc
        out = tf.nn.relu(out)
        return out