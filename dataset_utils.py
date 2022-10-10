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

'''
Utility file to manage batch generator for training and prepare input and labels to test on the FPGA board
'''

import os
import sys
import random
import shutil
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# directory where slices of all volumes are stored
target_dir = "build/dataset/target"
input_dir = "build/dataset/input"
extension = ".npy"

cal_samples = 500

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(extension) and int(fname.split(sep='-')[0]) > 20
    ]
)

target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".npy") and int(fname.split(sep='-')[0]) > 20
    ]
)

input_test_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(extension) and int(fname.split(sep='-')[0]) <= 20
    ]
)

target_test_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".npy") and int(fname.split(sep='-')[0]) <= 20
    ]
)

# Uncomment the following to check that input and labels slices are correctly paired
# print("Number of samples:", len(input_img_paths))
# print("Number of samples:", len(target_img_paths))
# for input_path, target_path in zip(input_img_paths[-10:], target_img_paths[-10:]):
#     print(input_path, "|", target_path)

for input_path, target_path in zip(input_img_paths, target_img_paths):
    assert os.path.basename(input_path) == os.path.basename(target_path)

for input_path, target_path in zip(input_test_img_paths, target_test_img_paths):
    assert os.path.basename(input_path) == os.path.basename(target_path)

random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)

random.Random(1337).shuffle(input_test_img_paths)
random.Random(1337).shuffle(target_test_img_paths)

val_samples = len(input_img_paths) // 5

train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]

val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

assert len(train_input_img_paths) == len(train_target_img_paths)
assert len(val_input_img_paths) == len(val_target_img_paths)


def explode_img(img, num_classes, img_size):
    """
    @param img: numpy array image to be transformed in a binary (n,n,num_classes) volume
    @param num_classes: the number of labels present in the dataset
    @param img_size: firsts two dimensions of the output 3D volume
    @return: A binary n by n by num_classes volume where img[:, :, class_i] is equal to 1 where pixel of the i-th
    class are present
    """
    exploded = np.zeros(shape=img_size + (num_classes,), dtype=np.uint8)
    for i in range(num_classes):
        exploded[:, :, i] = (img[:, :] == i).astype(np.uint8)
    return exploded


class DataGen(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_paths, target_paths, num_classes=6):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_paths
        self.target_img_paths = target_paths
        self.num_classes = num_classes

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size  # number of batches

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = np.load(path)
            x[j] = np.expand_dims(img, 2)
        y = np.zeros((self.batch_size,) + self.img_size + (self.num_classes,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = np.load(path)
            y[j] = explode_img(img, self.num_classes, self.img_size)  # the labels are presented as a binary volume
        return x, y

def get_DataGen4test(batch_size=8, img_size=(256, 256)):
    return DataGen(batch_size, img_size, input_test_img_paths, target_test_img_paths)

def get_train_len():
    return len(input_test_img_paths)

def get_DataGen(dataset="train", batch_size=64, img_size=(256, 256)):
    """
    Return a train-set or a validation-set or a calibration-set batches generator
    @param dataset: "train", "validation", "calibration", or "test"
    @param batch_size: number of images per batch.
    Default is 64
    @param img_size: dimension of a single image. Default (256, 256)
    @return: train-set if train is True and calibration False.
    Calibration-set if calibration is True and validation dataset if both are False
    """
    if dataset == "train":
        return DataGen(batch_size, img_size, train_input_img_paths, train_target_img_paths)
    elif dataset == "validation":
        return DataGen(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    elif dataset == "test":
        return DataGen(batch_size, img_size, input_test_img_paths, target_test_img_paths)
    elif dataset == "calibration":
        cal_input_img_paths = input_img_paths[-cal_samples:]
        cal_target_img_paths = target_img_paths[-cal_samples:]
        return DataGen(batch_size, img_size, cal_input_img_paths, cal_target_img_paths)
    else:
        raise ValueError('dataset should be equal to "train", "validation", "calibration", or "test"')


def prepare_target_images(start=0, num_images=1000):
    """
    Prepare directory with input and labels for evaluation on the board
    @param start: index of the first image to copy in the target folders
    @param num_images: number of images to include in both input and labels folders
    """
    os.makedirs('build/target/images', exist_ok=False)
    os.makedirs('build/target/labels', exist_ok=False)
    assert num_images + start < len(val_target_img_paths)
    for i in range(start, start + num_images):
        shutil.copy(input_test_img_paths[i],
                    'build/target/images/' + os.path.basename(os.path.normpath(input_test_img_paths[i])))
        shutil.copy(target_test_img_paths[i],
                    'build/target/labels/' + os.path.basename(os.path.normpath(target_test_img_paths[i])))
