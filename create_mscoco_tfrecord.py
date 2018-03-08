"""Converts MSCOCO data to TFRecord file format with Example protos.

The MSCOCO images are expected to reside in JPEG files located in the
directory structure as given by extracting the MSCOCO zip files:

  train2014/COCO_train2014_000000000151.jpg
  train2014/COCO_train2014_000000000260.jpg
  ...

and

  val2014/COCO_val2014_000000000042.jpg
  val2014/COCO_val2014_000000000073.jpg
  ...

where the year can be either 2014 or 2017.

The MSCOCO annotations JSON files are expected to reside in the directory
structure as given by extracting the MSCOCO zip files:

annotations/instances_train2014.json

and

annotations/instances_val2014.json

where the year can be either 2014 or 2017.

This script converts the combined MSCOCO data into sharded data files consisting
of 64, 4 and 0 TFRecord files, respectively:

  output_root/train-00000-of-00032
  output_root/train-00001-of-00032
  ...
  output_root/train-00063-of-00064

and

  output_root/val-00000-of-00004
  ...
  output_root/val-00003-of-00004

and

  output_root/test-00000-of-00000
  ...
  output_root/test-00000-of-00000

Each TFRecord file contains ~2000 records. Each record within the TFRecord file
is a serialized Example proto consisting of precisely one image and its associated
bounding box annotations.

NOTE: To use this script, you must install the  MSCOCO API by
executing the 'install_mscoco_api.sh' file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import before pycocotools to enable headless functionality.
import matplotlib
matplotlib.use('agg')

# Append to pycocotools to the current path inside Docker.
import os, sys
sys.path.append('/root/data/scripts/cocoapi/PythonAPI')

from pycocotools.coco import COCO
from PIL import Image
import random

import numpy as np
import tensorflow as tf
import logging

from datetime import datetime
import threading

import dataset_util

tf.flags.DEFINE_string("year", "2014", "MSCOCO dataset year.")

tf.flags.DEFINE_string("data_root", "/root/data/mscoco",
                       "Root directory to Microsoft COCO dataset.")

tf.flags.DEFINE_string("output_root", "/root/data/mscoco",
                       "Output data directory.")

tf.flags.DEFINE_bool("shuffle", True, "Toggle shuffling of data within dataset.")

tf.flags.DEFINE_integer("num_val", 8000,
                        "Number of image from val set to keep for validation.")

tf.flags.DEFINE_integer("train_shards", 64,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 4,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 0,
                        "Number of shards in testing TFRecord files.")

tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS


def load_mscoco_detection_dataset(img_dir, anno_path, shuffle_img=True):
  """Load data from dataset using pycocotools.

  Args:
      img_dir: directory of MSCOCO images
      anno_path: file path of coco annotations file
      shuffle_img: wheter to shuffle images order

  Returns:
      coco_data: list of dictionary-formatted data for each image
  """
  coco = COCO(anno_path)
  img_ids = coco.getImgIds()
  # [0,12,26,29,30,45,66,68,69,71,83] are missing (feature of MSCOCO dataset)
  cat_ids = coco.getCatIds()

  # Shuffle the ordering of images. Make the randomization repeatable.
  if shuffle_img:
    random.seed(12345)
    random.shuffle(img_ids)

  coco_data = []
  nb_imgs = len(img_ids)
  for index, img_id in enumerate(img_ids):
    if index % 100 == 0:
        print("Reading images: %d / %d "%(index, nb_imgs))

    img_info = {}
    bboxes = []
    labels = []

    img_detail = coco.loadImgs(img_id)[0]
    pic_height = img_detail['height']
    pic_width = img_detail['width']

    ann_ids = coco.getAnnIds(imgIds=img_id,catIds=cat_ids)
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
      bboxes_data = ann['bbox']
      # MSCOCO bounding boxes have format: [xmin, ymin, width, height]
      bboxes_data = [bboxes_data[0]/float(pic_width), \
                     bboxes_data[1]/float(pic_height),\
                     bboxes_data[2]/float(pic_width), \
                     bboxes_data[3]/float(pic_height)]

      bboxes.append(bboxes_data)
      labels.append(ann['category_id'])

    img_path = os.path.join(img_dir, img_detail['file_name'])
    img_bytes = tf.gfile.FastGFile(img_path,'rb').read()

    img_info['pixel_data'] = img_bytes
    img_info['height'] = pic_height
    img_info['width'] = pic_width
    img_info['bboxes'] = bboxes
    img_info['labels'] = labels

    coco_data.append(img_info)
  return coco_data


def _dict_to_example(data_dict):
  """Builds an Example proto for an image and its corresponding annotations.

  Args:
    data_dict: A python dictionary containing image and annotation data.

  Returns:
    An Example proto.
  """
  bboxes = data_dict['bboxes']
  xmin, xmax, ymin, ymax = [], [], [], []
  for bbox in bboxes:
    xmin.append(bbox[0])
    xmax.append(bbox[0] + bbox[2])
    ymin.append(bbox[1])
    ymax.append(bbox[1] + bbox[3])

  feature = tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(data_dict['height']),
    'image/width': dataset_util.int64_feature(data_dict['width']),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
    'image/object/class/label': dataset_util.int64_list_feature(data_dict['labels']),
    'image/encoded': dataset_util.bytes_feature(data_dict['pixel_data']),
    'image/format': dataset_util.bytes_feature('jpeg'.encode('utf-8')),
  })

  example = tf.train.Example(features=feature)
  return example


def _process_image_files(thread_index, ranges, name, datadict_list, num_shards):
  """Processes and saves a subset of datadict_list as TFRecord files in one thread.

  Args:
    thread_index: Integer thread identifier within [0, len(ranges)].
    ranges: A list of pairs of integers specifying the ranges of the dataset to
      process in parallel.
    name: Unique identifier specifying the dataset.
    datadict_list: List of dictionaries corresponding to elements in dataset.
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
  for s in xrange(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_root, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in images_in_shard:
      datadict = datadict_list[i]

      example = _dict_to_example(datadict)
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


def _process_dataset(name, datadict_list, num_shards):
  """Processes a complete data set and saves it as a TFRecord.

  Args:
    name: Unique identifier specifying the dataset.
    datadict_list: List of dictionaries corresponding to elements in dataset.
    num_shards: Integer number of shards for the output files.
  """

  # Break the datadict_list into num_threads batches. Batch i is defined as
  # datadict_list[ranges[i][0]:ranges[i][1]].
  num_threads = min(num_shards, FLAGS.num_threads)
  spacing = np.linspace(0, len(datadict_list), num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in xrange(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Launch a thread for each batch.
  print("Launching %d threads for spacings: %s" % (num_threads, ranges))
  for thread_index in xrange(len(ranges)):
    args = (thread_index, ranges, name, datadict_list, num_shards)
    t = threading.Thread(target=_process_image_files, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print("%s: Finished processing all %d images and annotations in data set '%s'." %
        (datetime.now(), len(datadict_list), name))


def main(unused_argv):
  valid_years = ["2014", "2017"]
  datasets = ["train", "val"]
  def _is_valid_num_shards(num_shards):
    """Returns True if num_shards is compatible with FLAGS.num_threads."""
    return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

  assert _is_valid_num_shards(FLAGS.train_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
  assert _is_valid_num_shards(FLAGS.val_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
  assert _is_valid_num_shards(FLAGS.test_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")

  assert FLAGS.year in valid_years, "Please supply a valid year."

  if not tf.gfile.IsDirectory(FLAGS.output_root):
    tf.gfile.MakeDirs(FLAGS.output_root)

  coco_data = {}
  for dataset in datasets:
    img_dir = os.path.join(FLAGS.data_root, dataset+FLAGS.year)
    anno_path = os.path.join(FLAGS.data_root, 'annotations',
                             'instances_{:s}.json').format(dataset+FLAGS.year)

    coco_data[dataset] = load_mscoco_detection_dataset(img_dir, anno_path,
                                                   shuffle_img=FLAGS.shuffle)

  # Redistribute the MSCOCO data.
  train_cutoff = int(len(coco_data['val'])-FLAGS.num_val)
  val_cutoff = int(len(coco_data['val']))

  train_dataset = coco_data['train'] + coco_data['val'][0:train_cutoff]
  val_dataset = coco_data['val'][train_cutoff:val_cutoff]
  test_dataset = coco_data['val'][val_cutoff:]

  _process_dataset("train", train_dataset, FLAGS.train_shards)
  _process_dataset("val", val_dataset, FLAGS.val_shards)
  _process_dataset("test", test_dataset, FLAGS.test_shards)

  print("Conversion complete.")


if __name__ == "__main__":
    tf.app.run()