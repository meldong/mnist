"""(R2U_KD) Converts image data to TFRecords file format with Example protos.

This module reads the images files and creates two TFRecord datasets:
one for train and one for test. Each TFRecord dataset is comprised of a set of
TF-Example protocol buffers, each of which contain a single image and label.
Adapted from the following source: https://github.com/tensorflow/models
/blob/master/research/slim/datasets/download_and_convert_flowers.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

import math
import os
import random
import sys

import tensorflow as tf

# The dataset directory where the dataset is stored.
_DATASET_DIR = '/tmp'

# The name of dataset.
_DATASET_NAME = 'flower'

# The number of images in the testing set.
_NUM_TEST = 350

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5


def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  """Returns a TF-Feature of floats.

  Args:
    values: A scalar of list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = '%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'test']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  dataset_root = os.path.join(dataset_dir, _DATASET_NAME + '_photos')
  directories = []
  class_names = []
  for filename in os.listdir(dataset_root):
    path = os.path.join(dataset_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'test'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'test']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  for shard_id in range(_NUM_SHARDS):
    output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
    writer = tf.python_io.TFRecordWriter(output_filename)

    start_ndx = shard_id * num_per_shard
    end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
    for i in range(start_ndx, end_ndx):
      sys.stdout.write('\r>> Converting image %d/%d shard %d'
                       % (i+1, len(filenames), shard_id))
      sys.stdout.flush()

      image = Image.open(filenames[i])
      image_data = image.tobytes()
      width, height = image.size

      class_name = os.path.basename(os.path.dirname(filenames[i]))
      class_id = class_names_to_ids[class_name]

      example = _image_to_tfexample(image_data, b'jpg', height, width, class_id)
      writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _write_label_file(labels_to_class_names, dataset_dir, filename='labels.md'):
  """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))


def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  tmp_dir = os.path.join(dataset_dir, _DATASET_NAME + '_photos')
  tf.gfile.DeleteRecursively(tmp_dir)


def main(_):
  """Runs the download and conversion operation."""
  if not tf.gfile.Exists(_DATASET_DIR):
    tf.gfile.MakeDirs(_DATASET_DIR)
    print('Dataset files do not exist. Put them in the dir %s.' % _DATASET_DIR)
    return

  if _dataset_exists(_DATASET_DIR):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  photo_filenames, class_names = _get_filenames_and_classes(_DATASET_DIR)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  training_filenames = photo_filenames[_NUM_TEST:]
  testing_filenames = photo_filenames[:_NUM_TEST]

  # First, convert the training and testing sets.
  _convert_dataset('train', training_filenames, class_names_to_ids,
                   _DATASET_DIR)
  _convert_dataset('test', testing_filenames, class_names_to_ids,
                   _DATASET_DIR)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  _write_label_file(labels_to_class_names, _DATASET_DIR)

  #_clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the dataset!')


if __name__ == '__main__':
  tf.app.run()
