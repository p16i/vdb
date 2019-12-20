import numpy as np

import tensorflow as tf

input_dims = {
    "mnist": (28, 28, 1),
    "fashion_mnist": (28, 28, 1),
    "cifar10": (32, 32, 3)
}

dataset_size = {
    "mnist": (60000, 10000),
    "fashion_mnist": (60000, 10000),
    "cifar10": (50000, 10000),
}

def subset_dataset_parsing(name):
    name, subset_size = name.split("-")

    return name, int(subset_size.replace("k", "000"))

def get_dataset(name, data_path="./datasets"):
    if name in [
            'cifar10-10k', 'cifar10-40k',
            'fashion_mnist-10k', 'fashion_mnist-40k'
        ]:

        subset_name = name
        name, subset_size = subset_dataset_parsing(subset_name)

        dataset_size[subset_name] = dataset_size[name]
        input_dims[subset_name] = input_dims[name]

        (train_images, train_labels), \
        (test_images, test_labels), \
        (selected_images, selected_labels) = get_dataset(name)

        chosen_ix = np.random.choice(
            range(train_images.shape[0]),
            subset_size,
            replace=False
        )

        train_images, train_labels = train_images[chosen_ix, :], train_labels[chosen_ix]

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

    ## Normalizing the images to the range of [-1, 1]
    if name in ["mnist", "fashion_mnist"]:
        train_images = 2*(train_images / 255.) - 1
        test_images  = 2*(test_images / 255.) - 1
    elif name == "cifar10":
        train_images = tf.image.per_image_standardization(train_images).numpy()
        test_images = tf.image.per_image_standardization(test_images).numpy()
    else:
        raise SystemError(f"No normalization implemented for {name}")

    data_ix = np.loadtxt(f"{data_path}/{name}_2d_samples").astype(int)

    selected_images = test_images[data_ix]
    selected_labels = test_labels[data_ix]

    return (train_images, train_labels), \
        (test_images, test_labels), \
        (selected_images, selected_labels)

def get_2d_samples(dataset):
    _, (test_images, test_labels), _ = get_dataset(dataset)

    data_ix = np.loadtxt(f"../datasets/{dataset}_2d_samples").astype(int)

    images = test_images[data_ix]
    labels = test_labels[data_ix]

    return (test_images, test_labels), (images, labels)