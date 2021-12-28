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
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import pickle


# TODO(jonycgn): Use tfc.PackedTensors once new binary packages have been built.
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


def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image


def write_png(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
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
    if img.size[0] >=256 and img.size[1] >= 256:
      img_files.append(imgfile)
  return img_files

def compress(args):
  """Compresses an image."""
  # Load input image and add batch dimension.
  x_input = tf.placeholder(dtype=tf.float32, shape=(1, None, None, 3))
  x = x_input
  weights_224_3c = tf.placeholder(tf.float32, [1, None, None, 3])
  weights_224_3c_label = tf.placeholder(tf.float32, [1, None, None, 3])
  x_attention = x * weights_224_3c
  x_attention_input = tf.concat([x, weights_224_3c, x_attention], axis=3)

  # Instantiate model.
  analysis_transform = AnalysisTransform(args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  synthesis_transform = SynthesisTransform(args.num_filters)

  # Transform and compress the image.
  y = analysis_transform(x_attention_input)
  string = entropy_bottleneck.compress(y)

  # Transform the quantized image back (if requested).
  y_hat, likelihoods = entropy_bottleneck(y, training=False)
  x_hat = synthesis_transform(y_hat)

  rec = synthesis_transform(y_hat)
  
  imgName = tf.placeholder(tf.string)
  op = write_png(imgName, rec[0,:,:,:])


  vgg = vgg16.Vgg16('/gdata/gaocs/pretrained_models/vgg16_no_fc.npy')
  vgg.build(x)
  feature_x = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]
  vgg.build(x_hat)
  feature_x_tilde = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]

  feature_x_mask = []
  feature_x_mask_invert = []
  for n in range(len(feature_x)):
    one = tf.ones_like(feature_x[n])
    zero = tf.zeros_like(feature_x[n])
    feat_mask = tf.where(feature_x[n]>0, x=one, y=zero)
    feature_x_mask.append(feat_mask)
    feature_x_mask_invert.append(feat_mask*(-1)+1)

  loss_feat_fore_all = []
  loss_feat_fore_sum = 0.0
  loss_feat_back_all = []
  loss_feat_back_sum = 0.0
  loss_feat_all = []
  loss_feat_sum = 0.0
  for n in range(len(feature_x)):
    loss_temp_fore = tf.reduce_mean(( (feature_x[n]-feature_x_tilde[n]) / (tf.reduce_mean(feature_x[n])+0.00000001) * feature_x_mask[n] )**2)
    loss_feat_fore_all.append(loss_temp_fore)
    loss_feat_fore_sum += loss_temp_fore

    loss_temp_back = tf.reduce_mean(( (feature_x[n]-feature_x_tilde[n]) / (tf.reduce_mean(feature_x[n])+0.00000001) * feature_x_mask_invert[n] )**2)
    loss_feat_back_all.append(loss_temp_back)
    loss_feat_back_sum += loss_temp_back

    loss_temp = tf.reduce_mean(( (feature_x[n]-feature_x_tilde[n]) / (tf.reduce_mean(feature_x[n])+0.00000001) )**2)
    loss_feat_all.append(loss_temp)
    loss_feat_sum += loss_temp
  loss_f_fore = loss_feat_fore_sum / len(feature_x) 
  loss_f_back = loss_feat_back_sum / len(feature_x)
  loss_f = loss_feat_sum / len(feature_x)

  num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

  # Total number of bits divided by number of pixels.
  eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Bring both images back to 0..255 range.
  x *= 255
  x_hat = tf.clip_by_value(x_hat, 0, 1)
  x_hat = tf.round(x_hat * 255)

  mse_foreground = tf.reduce_sum(tf.squared_difference(x*weights_224_3c, x_hat*weights_224_3c)) / tf.reduce_sum(weights_224_3c)
  psnr_foreground = 20 * tf.math.log(255.0 / tf.math.sqrt(mse_foreground)) / tf.math.log(10.0)
  msssim_foreground = tf.squeeze(tf.image.ssim_multiscale(x_hat*weights_224_3c, x*weights_224_3c, 255))
  weights_224_3c_invert = -1 * weights_224_3c + 1
  mse_background = tf.reduce_sum(tf.squared_difference(x*weights_224_3c_invert, x_hat*weights_224_3c_invert)) / tf.reduce_sum(weights_224_3c_invert)
  psnr_background = 20 * tf.math.log(255.0 / tf.math.sqrt(mse_background)) / tf.math.log(10.0)
  msssim_background = tf.squeeze(tf.image.ssim_multiscale(x_hat*weights_224_3c_invert, x*weights_224_3c_invert, 255))

  mse_fore = tf.reduce_sum(tf.squared_difference(x*weights_224_3c_label, x_hat*weights_224_3c_label)) / tf.reduce_sum(weights_224_3c_label)
  psnr_fore = 20 * tf.math.log(255.0 / tf.math.sqrt(mse_fore)) / tf.math.log(10.0)
  msssim_fore = tf.squeeze(tf.image.ssim_multiscale(x_hat*weights_224_3c_label, x*weights_224_3c_label, 255))
  weights_224_3c_label_invert = -1 * weights_224_3c_label + 1
  mse_back = tf.reduce_sum(tf.squared_difference(x*weights_224_3c_label_invert, x_hat*weights_224_3c_label_invert)) / tf.reduce_sum(weights_224_3c_label_invert)
  # mse_back = tf.reduce_sum(tf.squared_difference(x, x_hat)) / (tf.reduce_sum(weights_224_3c_label_invert) + tf.reduce_sum(weights_224_3c_label))
  psnr_back = 20 * tf.math.log(255.0 / tf.math.sqrt(mse_back)) / tf.math.log(10.0)
  msssim_back = tf.squeeze(tf.image.ssim_multiscale(x_hat*weights_224_3c_label_invert, x*weights_224_3c_label_invert, 255))

  mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
  psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
  msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

  with tf.Session() as sess:
    # Load the latest model checkpoint, get the compressed string and the tensor
    # shapes.
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    tensors = [string, tf.shape(x)[1:-1], tf.shape(y)[1:-1]]

    orgPath = args.OrgPath
    binPath = args.BinPath
    recPath = args.RecPath
    if not os.path.exists(binPath):
        os.mkdir(binPath)
    if not os.path.exists(recPath):
        os.mkdir(recPath)
    orgFiles = os.listdir(orgPath)
    orgFiles = sorted(orgFiles)
    # print(orgFiles)
    
    mse_foreground_all = []
    psnr_foreground_all = []
    msssim_foreground_all = []
    msssimdb_foreground_all = []
    loss_f_foreground_all = []
    loss_feat_foreground_list = []
    mse_background_all = []
    psnr_background_all = []
    msssim_background_all = []
    msssimdb_background_all = []
    loss_f_background_all = []
    loss_feat_background_list = []

    mse_fore_all = []
    psnr_fore_all = []
    msssim_fore_all = []
    msssimdb_fore_all = []
    mse_back_all = []
    psnr_back_all = []
    msssim_back_all = []
    msssimdb_back_all = []

    mse_all = []
    psnr_all = []
    msssim_all = []
    msssimdb_all = []
    eval_bpp_all = []
    bpp_all = []
    loss_f_all = []
    loss_feat_list = []

    pickle_name_label = '/gdata1/gaocs/pretrained_models/minVal2014_Test5000_0_1.pickle'
    with open (pickle_name_label, 'rb') as fp:
      print(pickle_name_label)
      all_weights_val5000_label = pickle.load(fp)
    
    pickle_name = '/gdata1/gaocs/pretrained_models/minVal2014_5000_Conv5_3_binary_dilation_0_1.pickle'
    with open (pickle_name, 'rb') as fp:
      print(pickle_name)
      all_weights_val5000 = pickle.load(fp)

    for idx, imgFile in enumerate(orgFiles):
    #   print(imgFile)
      img = Image.open(orgPath+imgFile)
      img = np.asarray(img, dtype=np.float32)
      if len(img.shape) != 3:
        # print(image_file)
        imgarr = np.zeros((img.shape[0],img.shape[1],3), dtype=np.float32)
        imgarr[:,:,0] = img
        imgarr[:,:,1] = img
        imgarr[:,:,2] = img
        img = np.expand_dims(imgarr, 0)
      else:
        img = np.expand_dims(img, 0)
      imgArr = img / 255

      pngRecName = recPath+imgFile[:-4]+'.png'
      weight_input_1c = all_weights_val5000[idx]   #[h,w]
      weight_input_3c = np.zeros((1, weight_input_1c.shape[0],weight_input_1c.shape[1],3), dtype=np.float32)
      for n in range(weight_input_3c.shape[3]):
        weight_input_3c[0,:,:,n] = weight_input_1c

      weight_input_1c_label = all_weights_val5000_label[idx]
      weight_input_3c_label = np.zeros((1, weight_input_1c_label.shape[0],weight_input_1c_label.shape[1],3), dtype=np.float32)
      for n in range(weight_input_3c_label.shape[3]):
        weight_input_3c_label[0,:,:,n] = weight_input_1c_label
      arrays = sess.run(tensors, feed_dict={x_input: imgArr, weights_224_3c:weight_input_3c, weights_224_3c_label:weight_input_3c_label, imgName:pngRecName})

      # Write a binary file with the shape information and the compressed string.
      packed = PackedTensors()
      packed.pack(tensors, arrays)
      # with open(binPath+imgFile[:-4]+'.bin', "wb") as f:
      #   f.write(packed.string)

      # If requested, transform the quantized image back and measure performance.
      if args.verbose:
        print(pngRecName)        
        # if not os.path.exists(pngRecName):
        eval_bpp_, mse_foreground_, mse_background_, mse_fore_, mse_back_, mse_, \
          psnr_foreground_, psnr_background_, psnr_fore_, psnr_back_, psnr_, \
          msssim_foreground_, msssim_background_, msssim_fore_, msssim_back_, msssim_, \
          num_pixels_, loss_f_fore_, loss_f_back_, loss_f_, \
          loss_feat_fore_, loss_feat_back_, loss_feat_, rec_, _ \
          = sess.run( [eval_bpp, mse_foreground, mse_background, mse_fore, mse_back, mse, \
            psnr_foreground, psnr_background, psnr_fore, psnr_back, psnr, \
            msssim_foreground, msssim_background, msssim_fore, msssim_back, msssim, \
            num_pixels, loss_f_fore, loss_f_back, loss_f, \
            loss_feat_fore_all, loss_feat_back_all, loss_feat_all, rec, op], feed_dict={x_input: imgArr, weights_224_3c:weight_input_3c, weights_224_3c_label:weight_input_3c_label, imgName:pngRecName})
        # else:
        #   eval_bpp_, mse_foreground_, mse_background_, mse_fore_, mse_back_, mse_, \
        #     psnr_foreground_, psnr_background_, psnr_fore_, psnr_back_, psnr_, \
        #     msssim_foreground_, msssim_background_, msssim_fore_, msssim_back_, msssim_, \
        #     num_pixels_, loss_f_fore_, loss_f_back_, loss_f_, \
        #     loss_feat_fore_, loss_feat_back_, loss_feat_, rec_\
        #     = sess.run( [eval_bpp, mse_foreground, mse_background, mse_fore, mse_back, mse, \
        #       psnr_foreground, psnr_background, psnr_fore, psnr_back, psnr, \
        #       msssim_foreground, msssim_background, msssim_fore, msssim_back, msssim, \
        #       num_pixels, loss_f_fore, loss_f_back, loss_f, \
        #       loss_feat_fore_all, loss_feat_back_all, loss_feat_all, rec], feed_dict={x_input: imgArr, weights_224_3c:weight_input_3c, weights_224_3c_label:weight_input_3c_label, imgName:pngRecName})

        # The actual bits per pixel including overhead.
        bpp = len(packed.string) * 8 / num_pixels_

        # if mse_foreground_ == 0:
        #   psnr_foreground_ = 60
        # if mse_fore == 0:
        #   psnr_fore = 60

        print("fore Mean squared error: {:0.4f}".format(mse_fore_))
        print("fore PSNR (dB): {:0.2f}".format(psnr_fore_))
        print("fore Multiscale SSIM: {:0.4f}".format(msssim_fore_))
        print("fore Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim_fore_)))
        print("back Mean squared error: {:0.4f}".format(mse_back_))
        print("back PSNR (dB): {:0.2f}".format(psnr_back_))
        print("back Multiscale SSIM: {:0.4f}".format(msssim_back_))
        print("back Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim_back_)))

        print("foreground Mean squared error: {:0.4f}".format(mse_foreground_))
        print("foreground PSNR (dB): {:0.2f}".format(psnr_foreground_))
        print("foreground Multiscale SSIM: {:0.4f}".format(msssim_foreground_))
        print("foreground Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim_foreground_)))
        print("foreground VGG loss: {:0.4f}".format(loss_f_fore_))
        np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
        print(loss_feat_fore_)

        print("background Mean squared error: {:0.4f}".format(mse_background_))
        print("background PSNR (dB): {:0.2f}".format(psnr_background_))
        print("background Multiscale SSIM: {:0.4f}".format(msssim_background_))
        print("background Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim_background_)))
        print("background VGG loss: {:0.4f}".format(loss_f_back_))
        np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
        print(loss_feat_back_)

        print("Mean squared error: {:0.4f}".format(mse_))
        print("PSNR (dB): {:0.2f}".format(psnr_))
        print("Multiscale SSIM: {:0.4f}".format(msssim_))
        print("Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim_)))
        print("Information content in bpp: {:0.4f}".format(eval_bpp_))
        print("Actual bits per pixel: {:0.4f}".format(bpp))
        print("VGG loss: {:0.4f}".format(loss_f_))
        np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
        print(loss_feat_)

        if mse_fore_ > 1e-8:
          mse_fore_all.append(mse_fore_)
          psnr_fore_all.append(psnr_fore_)
          msssim_fore_all.append(msssim_fore_)
          msssimdb_fore_all.append(-10*np.log10(1-msssim_fore_))
        if mse_back_ > 1e-8:
          mse_back_all.append(mse_back_)
          psnr_back_all.append(psnr_back_)
          msssim_back_all.append(msssim_back_)
          msssimdb_back_all.append(-10*np.log10(1-msssim_back_))
        if mse_foreground_ > 1e-8:
          mse_foreground_all.append(mse_foreground_)
          psnr_foreground_all.append(psnr_foreground_)
          msssim_foreground_all.append(msssim_foreground_)
          msssimdb_foreground_all.append(-10*np.log10(1-msssim_foreground_))
          loss_f_foreground_all.append(loss_f_fore_)
          loss_feat_foreground_list.append(loss_feat_fore_)
        if mse_background_ > 1e-8:
          mse_background_all.append(mse_background_)
          psnr_background_all.append(psnr_background_)
          msssim_background_all.append(msssim_background_)
          msssimdb_background_all.append(-10*np.log10(1-msssim_background_))
          loss_f_background_all.append(loss_f_back_)
          loss_feat_background_list.append(loss_feat_back_)

        mse_all.append(mse_)
        psnr_all.append(psnr_)
        msssim_all.append(msssim_)
        msssimdb_all.append(-10*np.log10(1-msssim_))
        eval_bpp_all.append(eval_bpp_)
        bpp_all.append(bpp)
        loss_f_all.append(loss_f_)
        np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
        loss_feat_list.append(loss_feat_)
    print('\n\n---total averege---')
    print("fore Mean squared error: {:0.4f}".format(np.mean(mse_fore_all)))
    print("fore PSNR (dB): {:0.2f}".format(np.mean(psnr_fore_all)))
    print("fore Multiscale SSIM: {:0.4f}".format(np.mean(msssim_fore_all)))
    print("fore Multiscale SSIM (dB): {:0.2f}".format(np.mean(msssimdb_fore_all)))
    print("back Mean squared error: {:0.4f}".format(np.mean(mse_back_all)))
    print("back PSNR (dB): {:0.2f}".format(np.mean(psnr_back_all)))
    print("back Multiscale SSIM: {:0.4f}".format(np.mean(msssim_back_all)))
    print("back Multiscale SSIM (dB): {:0.2f}".format(np.mean(msssimdb_back_all)))

    print("foreground Mean squared error: {:0.4f}".format(np.mean(mse_foreground_all)))
    print("foreground PSNR (dB): {:0.2f}".format(np.mean(psnr_foreground_all)))
    print("foreground Multiscale SSIM: {:0.4f}".format(np.mean(msssim_foreground_all)))
    print("foreground Multiscale SSIM (dB): {:0.2f}".format(np.mean(msssimdb_foreground_all)))
    print("foreground VGG loss: {:0.4f}".format(np.mean(loss_f_foreground_all )))
    np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
    print(np.mean(loss_feat_foreground_list , axis=0))
    print("background Mean squared error: {:0.4f}".format(np.mean(mse_background_all)))
    print("background PSNR (dB): {:0.2f}".format(np.mean(psnr_background_all)))
    print("background Multiscale SSIM: {:0.4f}".format(np.mean(msssim_background_all)))
    print("background Multiscale SSIM (dB): {:0.2f}".format(np.mean(msssimdb_background_all)))
    print("background VGG loss: {:0.4f}".format(np.mean(loss_f_background_all )))
    np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
    print(np.mean(loss_feat_background_list , axis=0))

    print("Mean squared error: {:0.4f}".format(np.mean(mse_all)))
    print("PSNR (dB): {:0.2f}".format(np.mean(psnr_all)))
    print("Multiscale SSIM: {:0.4f}".format(np.mean(msssim_all)))
    print("Multiscale SSIM (dB): {:0.2f}".format(np.mean(msssimdb_all)))
    print("Information content in bpp: {:0.4f}".format(np.mean(eval_bpp_all)))
    print("Actual bits per pixel: {:0.4f}".format(np.mean(bpp_all)))
    print("VGG loss: {:0.4f}".format(np.mean(loss_f_)))
    np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
    print(np.mean(loss_feat_list, axis=0))


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
      "--checkpoint_dir", default="train",
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
      help="Lambda for VGG tradefoff.")

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
        "OrgPath",
        help="OrgPath.")
    cmd.add_argument(
        "BinPath",
        help="BinPath.")
    cmd.add_argument(
        "RecPath",
        help="RecPath.")

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
    print('bls_compress_label_metric.py compress images!\n')
    print(args.checkpoint_dir)
    compress(args)
  elif args.command == "decompress":
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decompress(args)


if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
