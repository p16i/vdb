import numpy as np

import tensorflow as tf

from sklearn.model_selection import StratifiedShuffleSplit

input_dims = {
    "mnist": (28, 28, 1),
    "fashion_mnist": (28, 28, 1),
    "cifar10": (32, 32, 3),
    "cifar10-40k": (32, 32, 3)
}

dataset_size = {
    "mnist": (60000, 10000),
    "fashion_mnist": (60000, 10000),
    "cifar10": (50000, 10000),
    "cifar10-40k": (40000, 10000),
}

def get_dataset(name):
    if name == 'cifar10-40k':
        print(f"using subset: {name}")

        (train_images, train_labels), \
        (test_images, test_labels), \
        (selected_images, selected_labels) = get_dataset('cifar10')

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1 - 1.0*dataset_size['cifar10-40k'][0]/dataset_size['cifar10'][0]
        )

        # we don't care about the other ix
        chosen_ix, _  = next(sss.split(train_images, train_labels))
        train_images = train_images[chosen_ix, :, :]

        print(train_images.shape)
        train_labels = train_labels[chosen_ix]

        return (train_images, train_labels), \
            (test_images, test_labels), \
            (selected_images, selected_labels)


    ds_method = getattr(tf.keras.datasets, name)
    (train_images, train_labels), (test_images, test_labels) = ds_method.load_data()

    dim  = input_dims[name]
    train_images = train_images.reshape(train_images.shape[0], *dim)\
        .astype("float32")
    test_images = test_images.reshape(test_images.shape[0], *dim)\
        .astype("float32")

    train_labels = train_labels.astype("int32").reshape(-1)
    test_labels = test_labels.astype("int32").reshape(-1)

    if name in ["mnist", "fashion_mnist"]:
        ## Normalizing the images to the range of [-1, 1]
        train_images = 2*(train_images / 255.) - 1
        test_images  = 2*(test_images / 255.) - 1
    elif name is "cifar10":
        mean, std = (0.4912, 0.4818, 0.4460), (0.2470, 0.2434, 0.2614)
        train_images = (train_images/255. - mean) / std
        test_images = (test_images/255. - mean) / std

    ## Binarization
    # if not name == "cifar10":
    #     train_images[train_images >= .5] = 1.
    #     train_images[train_images < .5] = 0.
    #     test_images[test_images >= .5] = 1.
    #     test_images[test_images < .5] = 0.

    np.random.seed(101)
    indices = np.random.choice(test_labels.shape[0], 1000, replace=False)
    selected_labels = test_labels[indices]
    selected_images = test_images[indices, :]

    return (train_images, train_labels), \
        (test_images, test_labels), \
        (selected_images, selected_labels)
