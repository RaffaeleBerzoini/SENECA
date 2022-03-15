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

import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm

original_data_dir = 'OrganSegmentations/'
brain_label = 6
dataset_input_dir = '../build/dataset/input/'
dataset_target_dir = '../build/dataset/target/'
new_dim = 256


def gamma_transformation(img, high_in, high_out, low_in, low_out, gamma=3.5):
    return low_out + (high_out - low_out) * ((img - low_in) / (high_in - low_in)) ** gamma


def extract_slices():
    for i in tqdm(range(21, 140)):
        vol = nib.load(original_data_dir + f'volume-{i}.nii.gz').get_fdata()
        lab = nib.load(original_data_dir + f'labels-{i}.nii.gz').get_fdata()
        for z in range(vol.shape[-1]):
            slice_input = vol[:, :, z]
            label = lab[:, :, z]
            if not brain_label in np.rint(np.unique(label)).astype(np.int8):
                high_in = np.max(slice_input)
                low_in = np.min(slice_input)
                high_out = 1
                low_out = -1
                input_slice = gamma_transformation(slice_input, high_in, high_out, low_in, low_out)
                np.save(dataset_input_dir + f'{i}-{z}.npy', cv2.resize(input_slice, dsize=(new_dim, new_dim), interpolation=cv2.INTER_LINEAR))
                label = np.rint(label).astype(np.uint8)
                cv2.imwrite(dataset_target_dir + f'{i}-{z}.bmp', cv2.resize(label, dsize=(new_dim, new_dim), interpolation=cv2.INTER_LINEAR))


if __name__ == '__main__':
    extract_slices()
