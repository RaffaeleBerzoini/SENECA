# MIT License
#
# Copyright (c) 2022 Raffaele Berzoini, Eleonora D'Arnese, Davide Conficconi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

"""Utility file for loss and dice scores function. Used during floating point U-Net training"""

epsilon = 1e-5
smooth = 1
alpha = 0.7


# 0 background
# 1 liver
# 2 bladder
# 3 lungs
# 4 kidneys
# 5 bone

def tversky_liver(y_true, y_pred, num_channel=1):
    true_pos = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel])
    false_neg = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]))
    false_pos = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel])
    return K.mean((true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth))


def tversky_bladder(y_true, y_pred, num_channel=2):
    true_pos = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel])
    false_neg = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]))
    false_pos = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel])
    return K.mean((true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth))


def tversky_lungs(y_true, y_pred, num_channel=3):
    true_pos = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel])
    false_neg = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]))
    false_pos = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel])
    return K.mean((true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth))


def tversky_kidneys(y_true, y_pred, num_channel=4):
    true_pos = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel])
    false_neg = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]))
    false_pos = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel])
    return K.mean((true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth))


def tversky_bones(y_true, y_pred, num_channel=5):
    true_pos = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel])
    false_neg = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]))
    false_pos = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel])
    return K.mean((true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth))


# 0 background
# 1 liver
# 2 bladder
# 3 lungs
# 4 kidneys
# 5 bones

# Empirical weights used for a more balanced training between organs with different frequency
background_w = 0.0
liver_w = 1.15
bladder_w = 1.95
lungs_w = 1
kidneys_w = 1.55
bones_w = 1
weights_sum = liver_w + bladder_w + lungs_w + kidneys_w + bones_w


def tversky_index(y_true, y_pred):
    return ((liver_w * tversky_liver(y_true, y_pred) + bladder_w * tversky_bladder(y_true, y_pred) +
             lungs_w * tversky_lungs(y_true, y_pred) + kidneys_w * tversky_kidneys(y_true, y_pred) +
             bones_w * tversky_bones(y_true, y_pred)) / weights_sum)


def foc_tversky_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    pt_2 = tversky_index(y_true, y_pred)
    gamma = 4.0 / 3.0
    return K.pow((1 - pt_2), gamma)


def dice_background(y_true, y_pred, smooth=1, num_class=0):
    intersection = K.sum(y_true[:, :, :, num_class] * y_pred[:, :, :, num_class])
    union = K.sum(y_true[:, :, :, num_class]) + K.sum(y_pred[:, :, :, num_class])
    return K.mean((2. * intersection + smooth) / (union + smooth))


def dice_liver(y_true, y_pred, smooth=1, num_class=1):
    intersection = K.sum(y_true[:, :, :, num_class] * y_pred[:, :, :, num_class])
    union = K.sum(y_true[:, :, :, num_class]) + K.sum(y_pred[:, :, :, num_class])
    return K.mean((2. * intersection + smooth) / (union + smooth))


def dice_bladder(y_true, y_pred, smooth=1, num_class=2):
    intersection = K.sum(y_true[:, :, :, num_class] * y_pred[:, :, :, num_class])
    union = K.sum(y_true[:, :, :, num_class]) + K.sum(y_pred[:, :, :, num_class])
    return K.mean((2. * intersection + smooth) / (union + smooth))


def dice_lungs(y_true, y_pred, smooth=1, num_class=3):
    intersection = K.sum(y_true[:, :, :, num_class] * y_pred[:, :, :, num_class])
    union = K.sum(y_true[:, :, :, num_class]) + K.sum(y_pred[:, :, :, num_class])
    return K.mean((2. * intersection + smooth) / (union + smooth))


def dice_kidneys(y_true, y_pred, smooth=1, num_class=4):
    intersection = K.sum(y_true[:, :, :, num_class] * y_pred[:, :, :, num_class])
    union = K.sum(y_true[:, :, :, num_class]) + K.sum(y_pred[:, :, :, num_class])
    return K.mean((2. * intersection + smooth) / (union + smooth))


def dice_bones(y_true, y_pred, smooth=1, num_class=5):
    intersection = K.sum(y_true[:, :, :, num_class] * y_pred[:, :, :, num_class])
    union = K.sum(y_true[:, :, :, num_class]) + K.sum(y_pred[:, :, :, num_class])
    return K.mean((2. * intersection + smooth) / (union + smooth))


# 0 background
# 1 liver
# 2 bladder
# 3 lungs
# 4 kidneys
# 5 bone

# Weights have been extracted by counting how many pixels of each organ were present in the whole dataset Weights are
# not computed at runtime just for training time purposes. The evaluate.py script compute the exacts weights for the
# evaluated dataset
def dice(y_true, y_pred, smooth=1):
    return ((0.23212333520332026 * dice_liver(y_true, y_pred) + 0.04549370195613813 * dice_bladder(y_true, y_pred)
             + 0.37348887454707363 * dice_lungs(y_true, y_pred) + 0.05246318852416101 * dice_kidneys(y_true, y_pred)
             + 0.2964308997693069 * dice_bones(y_true, y_pred)) / (0.23212333520332026 + 0.04549370195613813 +
                                                                   0.37348887454707363 + 0.05246318852416101 +
                                                                   0.2964308997693069))


def dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    return 1 - dice(y_true, y_pred)
