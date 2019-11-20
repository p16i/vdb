import tensorflow as tf

def log_metrics(writer, labels, metrics, step):
    with writer.as_default():
        for l, m in zip(labels, metrics):
            tf.summary.scalar(l, m, step=step)

# taken from https://stackoverflow.com/posts/38676842/revisions
def summary_image(writer, buffer, name, epoch):
    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    with writer.as_default():
        tf.summary.image(name, image, step=epoch)

