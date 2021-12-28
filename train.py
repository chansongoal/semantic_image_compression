# -*- coding: utf-8 -*-
# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic nonlinear transform coder for RGB images.

This is a close approximation of the image compression model published in:
J. Ballé, V. Laparra, E.P. Simoncelli (2017):
"End-to-end Optimized Image Compression"
Int. Conf. on Learning Representations (ICLR), 2017
https://arxiv.org/abs/1611.01704

With patches from Victor Xing <victor.t.xing@gmail.com>

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import sys

from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow as tf

import tensorflow_compression as tfc
from PIL import Image
import vgg16
import custom_vgg16
import os
import random
import h5py

class PackedTensors(object):
  """Packed representation of compressed tensors."""

  def __init__(self, string=None):
    self._example = tf.train.Example()
    if string:
      self.string = string

  @property
  def model(self):
    """Model identifier."""
    buf = self._example.features.feature["MD"].bytes_list.value[0]
    return buf.decode("ascii")

  @model.setter
  def model(self, value):
    self._example.features.feature["MD"].bytes_list.value[:] = [
        value.encode("ascii")]

  @model.deleter
  def model(self):
    del self._example.features.feature["MD"]

  @property
  def string(self):
    """A string representation of this object."""
    return self._example.SerializeToString()

  @string.setter
  def string(self, value):
    self._example.ParseFromString(value)

  def pack(self, tensors, arrays):
    """Packs Tensor values into this object."""
    if len(tensors) != len(arrays):
      raise ValueError("`tensors` and `arrays` must have same length.")
    i = 1
    for tensor, array in zip(tensors, arrays):
      feature = self._example.features.feature[chr(i)]
      feature.Clear()
      if array.ndim != 1:
        raise RuntimeError("Unexpected tensor rank: {}.".format(array.ndim))
      if tensor.dtype.is_integer:
        feature.int64_list.value[:] = array
      elif tensor.dtype == tf.string:
        feature.bytes_list.value[:] = array
      else:
        raise RuntimeError(
            "Unexpected tensor dtype: '{}'.".format(tensor.dtype))
      i += 1
    # Delete any remaining, previously set arrays.
    while chr(i) in self._example.features.feature:
      del self._example.features.feature[chr(i)]
      i += 1

  def unpack(self, tensors):
    """Unpacks Tensor values from this object."""
    arrays = []
    for i, tensor in enumerate(tensors):
      feature = self._example.features.feature[chr(i + 1)]
      np_dtype = tensor.dtype.as_numpy_dtype
      if tensor.dtype.is_integer:
        arrays.append(np.array(feature.int64_list.value, dtype=np_dtype))
      elif tensor.dtype == tf.string:
        arrays.append(np.array(feature.bytes_list.value, dtype=np_dtype))
      else:
        raise RuntimeError(
            "Unexpected tensor dtype: '{}'.".format(tensor.dtype))
    return arrays


def read_png(filename):
  """Loads a PNG image file."""
  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image

def read_img(filename):
  """Loads a image file."""
  img = Image.open(filename)
  arr = np.asarray(img)
  img.close()
  image = tf.cast(arr, tf.float32)
  image /= 255
  return image

def quantize_image(image):
  image = tf.round(image * 255)
  a = image
  image = tf.saturate_cast(image, tf.uint8)
  return image

def write_png(filename, image):
  """Saves an image to a PNG file."""
  # print(filename)
  image = quantize_image(image)
  # print(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)

class AnalysisTransform(tf.keras.layers.Layer):
  """The analysis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(AnalysisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (9, 9), name="layer_0", corr=True, strides_down=4,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_0")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_1")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=False,
            activation=None),
    ]
    super(AnalysisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class SynthesisTransform(tf.keras.layers.Layer):
  """The synthesis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(SynthesisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_0", inverse=True)),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_1", inverse=True)),
        tfc.SignalConv2D(
            3, (9, 9), name="layer_2", corr=False, strides_up=4,
            padding="same_zeros", use_bias=True,
            activation=None),
    ]
    super(SynthesisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor

def preprocess_image_files(image_files):
  img_files = []
  for idx, imgfile in enumerate(image_files):
    img = Image.open(imgfile)
    if img.size[0] >=256 and img.size[1] >= 255:
      img_files.append(imgfile)
  return img_files


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def normalize(recmax, recmin, orgmax, orgmin, data):
    return ((recmax-recmin)*(data-orgmin)/(orgmax-orgmin)+recmin)

def train(args):
  """Trains the model."""
  if args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

  num_pixels = args.batchsize * args.patchsize ** 2

  # Get training patch from dataset.
  x = tf.placeholder(tf.float32, [None, args.patchsize, args.patchsize, 3])
  weights_224_3c = tf.placeholder(tf.float32, [None, args.patchsize, args.patchsize, 3])
  weights_224_1c = tf.placeholder(tf.float32, [None, 224, 224])

  x_attention = x * weights_224_3c
  x_attention_input = tf.concat([x, weights_224_3c, x_attention], axis=3)

  weights_224_reshaped = tf.transpose(weights_224_1c, [1,2,0])
  weights_14_reshaped = tf.image.resize(weights_224_reshaped, [14,14], method='nearest')
  weights_28_reshaped = tf.image.resize(weights_224_reshaped, [28,28], method='nearest')
  weights_56_reshaped = tf.image.resize(weights_224_reshaped, [56,56], method='nearest')
  weights_112_reshaped = tf.image.resize(weights_224_reshaped, [112,112], method='nearest')

  weights_14 = tf.transpose(weights_14_reshaped, [2,0,1])
  weights_28 = tf.transpose(weights_28_reshaped, [2,0,1])
  weights_56 = tf.transpose(weights_56_reshaped, [2,0,1])
  weights_112 = tf.transpose(weights_112_reshaped, [2,0,1])

  all_weights = [weights_224_1c, weights_112, weights_56, weights_28, weights_14]

  # Instantiate model.
  analysis_transform = AnalysisTransform(args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  synthesis_transform = SynthesisTransform(args.num_filters)

  # Build autoencoder.
  # y = analysis_transform(x)
  y = analysis_transform(x_attention_input)

  y_tilde, likelihoods = entropy_bottleneck(y, training=True)
  x_tilde = synthesis_transform(y_tilde)    #[None, 224,224,3]

  vgg = vgg16.Vgg16('/gdata/gaocs/pretrained_models/vgg16_no_fc.npy')
  vgg.build(x)
  feature_x = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]
  vgg.build(x_tilde)
  feature_x_tilde = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]

  loss_feat_all = []
  loss_feat_sum = 0.0
  for n in range(len(feature_x)):
    f = tf.transpose(feature_x[n], [3,0,1,2])
    f_ = tf.transpose(feature_x_tilde[n], [3,0,1,2])
    loss_temp = tf.reduce_mean(( (f-f_) / (tf.reduce_mean(f)+0.00000001) )**2)
    loss_feat_all.append(loss_temp)
    loss_feat_sum += loss_temp
  loss_f = loss_feat_sum / len(feature_x)

  # Total number of bits divided by number of pixels.
  train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  # train_mse = tf.reduce_mean(tf.squared_difference(x,x_tilde))

  x_transposed = tf.transpose(x, [3,0,1,2])
  x_tilde_transposed = tf.transpose(x_tilde, [3,0,1,2])
  train_se = tf.squared_difference(x_transposed,x_tilde_transposed)
  weights_224_1c_plus = weights_224_1c + 0.1
  train_mse = tf.reduce_mean(train_se * weights_224_1c_plus)
  # train_mse = tf.reduce_mean(tf.squared_difference(x_transposed*weights_224_1c,x_tilde_transposed*weights_224_1c))


  # Multiply by 255^2 to correct for rescaling.
  train_mse *= 255 ** 2

  # The rate-distortion cost.
  # train_loss = args.lmbda_mse * train_mse + train_bpp
  # train_loss = args.lmbda_vgg * loss_f + train_bpp
  train_loss = args.lmbda_mse*train_mse + args.lmbda_vgg*loss_f + train_bpp

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

  tf.summary.scalar("loss", train_loss)
  tf.summary.scalar("bpp", train_bpp)
  tf.summary.scalar("mse", train_mse)
  tf.summary.scalar("vgg_avg", loss_f)
  tf.summary.scalar("vgg_5", loss_feat_all[-1])

  tf.summary.image("original", quantize_image(x))
  tf.summary.image("reconstruction", quantize_image(x_tilde))

  hooks = [
      tf.train.StopAtStepHook(last_step=args.last_step),
      tf.train.NanTensorHook(train_loss),
  ]

  with tf.device("/cpu:0"):
    train_dataset = np.load('/gdata/gaocs/dataset/COCO/train2014_224_80000.npy')
    all_weights_train = np.load('/gdata1/gaocs/pretrained_models/LabelAttention_Train_80000_0_1.npy')
    val_dataset = np.load('/gdata/gaocs/dataset/COCO/val2014_224_16000.npy')
    all_weights_val = np.load('/gdata1/gaocs/pretrained_models/LabelAttention_Val_16000_0_1.npy') 


  with tf.train.MonitoredTrainingSession(
      hooks=hooks, checkpoint_dir=args.checkpoint_dir,
      save_checkpoint_secs=300,  \
      config=tf.ConfigProto(log_device_placement=False) ) as sess:
    train_step = 0
    while not sess.should_stop():
      if train_step < int(train_dataset.shape[0] / args.batchsize):
        x_input = train_dataset[train_step*args.batchsize:(train_step+1)*args.batchsize, :,:,:] / 255.0
        weight_input_1c = all_weights_train[train_step*args.batchsize:(train_step+1)*args.batchsize,:,:] 
        weight_input_3c = np.zeros((args.batchsize, 224,224,3), dtype=np.float32)
        for n in range(weight_input_3c.shape[3]):
          weight_input_3c[:,:,:,n] = weight_input_1c
        sess.run(train_op, feed_dict={x:x_input, weights_224_1c:weight_input_1c, weights_224_3c:weight_input_3c})
        train_step += 1
        if train_step % int(train_dataset.shape[0]/args.batchsize/4) == 0:
          train_bpp_all = []
          train_mse_all = []
          loss_f_all = []
          loss_feat_list = []
          loss_total_all = []

          for n in range(int(val_dataset.shape[0]/args.batchsize/4)):
            x_input = val_dataset[n*args.batchsize:(n+1)*args.batchsize,:,:,:] / 255.0
            weight_input_1C = all_weights_val[n*args.batchsize:(n+1)*args.batchsize,:,:]
            weight_input_3C = np.zeros((args.batchsize, 224,224,3), dtype=np.float32)
            for n in range(weight_input_3C.shape[3]):
              weight_input_3C[:,:,:,n] = weight_input_1C
            train_bpp_, train_mse_, loss_f_, loss_feat_, loss_total = sess.run([train_bpp, train_mse, loss_f, loss_feat_all, train_loss], feed_dict={x:x_input, weights_224_1c:weight_input_1c, weights_224_3c:weight_input_3c})
            
            train_bpp_all.append(train_bpp_)
            train_mse_all.append(train_mse_)
            loss_f_all.append(loss_f_)
            loss_feat_list.append(loss_feat_)
            loss_total_all.append(loss_total)
          
          print(np.mean(train_bpp_all), np.mean(train_mse_all), np.mean(loss_f_all), np.mean(loss_total_all))
          feature_loss = np.mean(loss_feat_list, axis=0)
          print(feature_loss)

          loss_file = open('/ghome/gaocs/compression-master/examples/round1/mse_label_vgg_B_ml/1/eval_loss_mse_label_vgg_B_ml_1_' + str(args.lmbda_mse)+'_'+str(args.lmbda_vgg)+'.log', mode='a')
          loss_file.write(str(np.mean(train_bpp_all)))
          loss_file.write(' ')
          loss_file.write(str(np.mean(train_mse_all)))
          loss_file.write(' ')
          loss_file.write(str(np.mean(loss_f_all)))
          loss_file.write(' ')
          loss_file.write(str(np.mean(loss_total_all)))
          loss_file.write('\n')
          for n in range(feature_loss.shape[0]):
            loss_file.write(str(feature_loss[n]))
            loss_file.write(' ')
          loss_file.write('\n')
          loss_file.close()
      else:
        train_step = 0

def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--num_filters", type=int, default=128,
      help="Number of filters per layer.")
  parser.add_argument(
      "--checkpoint_dir", default="../../models/m0.001_v0.01_s500000/",
      help="Directory where to save/load model checkpoints.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model.")

  parser.add_argument(
      "--train_glob", default="images/*.jpg",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format.")

  parser.add_argument(
      "--batchsize", type=int, default=32,
      help="Batch size for training.")

  parser.add_argument(
      "--patchsize", type=int, default=224,
      help="Size of image patches for training.")

  parser.add_argument(
      "--lambda_mse", type=float, default=0.01, dest="lmbda_mse",
      help="Lambda for MSE tradeoff.")
  parser.add_argument(
      "--lambda_vgg", type=float, default=1, dest="lmbda_vgg",
      help="Lambda for VGG tradeoff.")
  

  parser.add_argument(
      "--last_step", type=int, default=2000,
      help="Train up to this number of steps.")

  parser.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")

  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes a TFCI file.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image, and writes back "
                  "a PNG file.")

  # Arguments for both 'compress' and 'decompress'.
  for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
    cmd.add_argument(
        "input_file",
        help="Input filename.")
    cmd.add_argument(
        "output_file", nargs="?",
        help="Output filename (optional). If not provided, appends '{}' to "
             "the input filename.".format(ext))
  # for cmd, ext in ((train_cmd, ".tfci"), (compress_cmd, ".png")):
  #   cmd.add_argument(
  #       "OrgPath",
  #       help="OrgPath.")
  #   cmd.add_argument(
  #       "BinPath",
  #       help="BinPath.")
  #   cmd.add_argument(
  #       "RecPath",
  #       help="RecPath.")

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Invoke subcommand.
  if args.command == "train":
    train(args)
  elif args.command == "compress":
    compress(args)

  elif args.command == "decompress":
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decompress(args)


if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
