"""Convert PascalVOC format to TFRecord."""
import os
import sys
import threading

from lxml import etree
import hashlib
import io
from PIL import Image

from datetime import datetime
import numpy as np
import tensorflow as tf

sys.path.append('/home/lukealexmiller/cartwatch/cartwatch-preseeding/code/models/research')

from object_detection.utils import dataset_util

tf.flags.DEFINE_string("output_root", "/home/lukealexmiller/tmp/HollywoodHeads",
                       "Output data directory.")

tf.flags.DEFINE_integer("img_maxside", 1024, "Maximum side of image saved to TF Record.")

tf.flags.DEFINE_integer("train_shards", 8,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 1,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 1,
                        "Number of shards in testing TFRecord files.")

tf.flags.DEFINE_integer("num_threads", 4,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS

class_map = {
    1: 'head'
}

class_index_map = {v: k for k, v in class_map.items()}


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.
  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.
  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.
  Returns:
    example: The converted tf.Example.
  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []

  for obj in data['object']:
    if isinstance(obj, str):
      continue

    difficult = bool(int(obj.get('difficult', 0)))
    if ignore_difficult_instances and difficult:
      continue

    difficult_obj.append(int(difficult))

    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])
    truncated.append(int(obj.get('truncated', 0)))
    poses.append(obj.get('pose', '').encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def _process_image_files(thread_index, ranges, name, exports, num_shards):
    """Processes and saves a subset of exports as TFRecord files in one thread.
    Args:
    thread_index: Integer thread identifier within [0, len(ranges)].
    ranges: A list of pairs of integers specifying the ranges of the dataset to
      process in parallel.
    name: Unique identifier specifying the dataset.
    exports: List of dictionaries corresponding to elements in dataset.
    num_shards: Integer number of shards for the output files.
    """
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128, and num_threads = 2, then the first thread
    # would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_root, 'TFRecords', output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in images_in_shard:
            path = os.path.join(FLAGS.output_root, 'Annotations', exports[i] + '.xml')

            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()

            xml = etree.fromstring(xml_str)

            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            example = dict_to_tf_example(data, FLAGS.output_root, class_index_map)

            if example is not None:
                writer.write(example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                      (datetime.now(), thread_index, counter, num_images_in_thread))
                sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d images with annotations to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
        print("%s [thread %d]: Wrote %d images with annotations to %d shards." %
              (datetime.now(), thread_index, counter, num_shards_per_batch))
        sys.stdout.flush()


def _process_dataset(name, exports, num_shards):
    """Processes a complete data set and saves it as a TFRecord.
    Args:
      name: Unique identifier specifying the dataset.
      exports: List of dictionaries corresponding to elements in dataset.
      num_shards: Integer number of shards for the output files.
    """

    # Break the exports into num_threads batches. Batch i is defined as
    # exports[ranges[i][0]:ranges[i][1]].
    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(exports), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, exports, num_shards)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d images and annotations in data set '%s'." %
          (datetime.now(), len(exports), name))


def main(unused_argv):
    def _is_valid_num_shards(num_shards):
        """Returns True if num_shards is compatible with FLAGS.num_threads."""
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert _is_valid_num_shards(FLAGS.train_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
    assert _is_valid_num_shards(FLAGS.val_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
    assert _is_valid_num_shards(FLAGS.test_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")

    if not tf.gfile.IsDirectory(os.path.join(FLAGS.output_root, 'TFRecords')):
        tf.gfile.MakeDirs(os.path.join(FLAGS.output_root, 'TFRecords'))

    for split_set in ('val', 'test', 'train'):

        split_filename = split_set + '.txt'
        split_filepath = os.path.join(FLAGS.output_root, 'Splits', split_filename)

        f = open(split_filepath, 'r')
        filename_images = f.read().splitlines()

        if split_set == 'train':
            _process_dataset(split_set, filename_images, FLAGS.train_shards)
        if split_set == 'val':
            _process_dataset(split_set, filename_images, FLAGS.val_shards)
        if split_set == 'test':
            _process_dataset(split_set, filename_images, FLAGS.test_shards)


if __name__ == '__main__':
    tf.app.run()
