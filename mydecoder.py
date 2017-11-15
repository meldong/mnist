import tensorflow as tf
import os.path

def read_and_decode(filename_queue, image_pixels):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image/encoded': tf.FixedLenFeature([], tf.string),
          'image/class/label': tf.FixedLenFeature([], tf.int64),
      })

  # Convert from a scalar string tensor (whose single string has
  # length IMAGE_PIXELS) to a uint8 tensor with shape
  # [IMAGE_PIXELS].
  image = tf.decode_raw(features['image/encoded'], tf.uint8)
  image.set_shape([image_pixels])

  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['image/class/label'], tf.int32)

  return image, label


def image_preprocessing_fn(image):
  """ to be here """
  return image


def inputs(train_dir, train, batch_size, num_epochs, image_pixels):
  """Reads input data num_epochs times.

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, num_classes).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None
  filename = os.path.join(train_dir, 'train' if train else 'test')
  filename = tf.gfile.Glob(filename + '*')
  tf.logging.info('Reading data from %s', filename)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        filename, num_epochs=num_epochs, shuffle=True)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue, image_pixels)

    # Pre-process the image
    image = image_preprocessing_fn(image)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return images, sparse_labels
