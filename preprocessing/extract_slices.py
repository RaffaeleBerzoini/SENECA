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
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm
from preprocess_volumes import *

original_data_dir = 'OrganSegmentations/'
same_vox_data_dir = 'OrganSegmentationsSameVoxel/'
brain_label = 6
dataset_input_dir = '../build/dataset/input/'
dataset_target_dir = '../build/dataset/target/'
new_dim = (256, 256)


def gamma_transformation(img, high_in, high_out, low_in, low_out, gamma=3.5):
    img[img < low_in] = low_in
    img[img > high_in] = high_in
    return low_out + ((high_out - low_out) * ((img - low_in + 1e-5) / (high_in - low_in + 1e-5)) ** gamma)


def remove_background(slice):
    n_pixel = slice.shape[0] * slice.shape[1]
    slice4hist = slice.copy()
    hist = plt.hist(slice4hist.flatten(), bins=100)
    i = 1
    while np.sum(hist[0][:i]) / n_pixel < 0.3:
        i = i + 1

    threshold = hist[1][i]
    slice[slice < threshold] = np.min(slice)

    return slice


def extract_slices():
    vox_size_out, _ = volumes_analysis()
    print("All voxels of all volumes will be scaled to: ", vox_size_out)

    for i in tqdm(range(140)):
        vol = adjust_voxel_size(original_data_dir + f'volume-{i}.nii.gz', vox_size_out=vox_size_out)
        lab = adjust_voxel_size(original_data_dir + f'labels-{i}.nii.gz', vox_size_out=vox_size_out, isLabel=True)

        # UNCOMMENT TO SAVE VOX_ADJUSTED_VOLS
        # nib.save(vol, same_vox_data_dir + f'volume-{i}.nii.gz')
        # nib.save(lab, same_vox_data_dir + f'labels-{i}.nii.gz')

        vol = vol.get_fdata()
        lab = lab.get_fdata()

        # vol = nib.load(original_data_dir + f'volume-{i}.nii.gz').get_fdata()
        # lab = nib.load(original_data_dir + f'labels-{i}.nii.gz').get_fdata()

        # print(f'Saving {vol.shape[-1]} slices...')
        for z in range(vol.shape[-1]):
            slice_input = vol[:, :, z]
            # slice_input = remove_background(slice_input)
            label = lab[:, :, z]
            if not brain_label in np.rint(np.unique(label)).astype(np.int8):
                high_in = 1000 # np.max(slice_input)
                low_in = -500 # np.min(slice_input)
                high_out = 1
                low_out = -1
                input_slice = gamma_transformation(slice_input, high_in, high_out, low_in, low_out, gamma=1)
                np.save(dataset_input_dir + f'{i}-{z}.npy', cv2.resize(input_slice, dsize=new_dim))
                label = np.rint(label).astype(np.uint8)
                np.save(dataset_target_dir + f'{i}-{z}.npy', cv2.resize(label, dsize=new_dim))
                # print(dataset_input_dir + f'{i}-{z}.npy', ' ', dataset_target_dir + f'{i}-{z}.npy')


if __name__ == '__main__':
    extract_slices()
