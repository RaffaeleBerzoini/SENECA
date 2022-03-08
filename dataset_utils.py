import os
import sys
import random
import shutil
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

target_dir = "build/dataset/target"
input_dir = "build/dataset/input"
extension = ".npy"

cal_samples = 500

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(extension)
    ]
)

target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".bmp")
    ]
)

# Uncomment the following to check that input and labels slices are correctly paired
# print("Number of samples:", len(input_img_paths))
# print("Number of samples:", len(target_img_paths))
# for input_path, target_path in zip(input_img_paths[-10:], target_img_paths[-10:]):
#     print(input_path, "|", target_path)

random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)

val_samples = len(input_img_paths) // 5

train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]

val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

assert len(train_input_img_paths) == len(train_target_img_paths)
assert len(val_input_img_paths) == len(val_target_img_paths)


def explode_img(img, num_classes, img_size):
    exploded = np.zeros(shape=img_size + (num_classes,), dtype=np.uint8)
    for i in range(num_classes):
        exploded[:, :, i] = (img[:, :, 0] == i).astype(np.uint8)
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
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            img = keras.preprocessing.image.img_to_array(img)
            y[j] = explode_img(img, self.num_classes, self.img_size)
        return x, y


def get_DataGen(train=True, batch_size=64, img_size=(256, 256), calibration=False):
    if calibration is True:
        cal_input_img_paths = input_img_paths[-cal_samples:]
        cal_target_img_paths = target_img_paths[-cal_samples:]
        return DataGen(batch_size, img_size, cal_input_img_paths, cal_target_img_paths)
    if train is True:
        return DataGen(batch_size, img_size, train_input_img_paths, train_target_img_paths)
    return DataGen(batch_size, img_size, val_input_img_paths, val_target_img_paths)


def prepare_target_images(start=0, num_images=1000):
    os.makedirs('build/target/images', exist_ok=False)
    os.makedirs('build/target/labels', exist_ok=False)
    assert num_images <= len(val_input_img_paths)
    for i in range(start, start + num_images):
        shutil.copy(val_input_img_paths[i],
                    'build/target/images/' + os.path.basename(os.path.normpath(val_input_img_paths[i])))
        shutil.copy(val_target_img_paths[i],
                    'build/target/labels/' + os.path.basename(os.path.normpath(val_target_img_paths[i])))
