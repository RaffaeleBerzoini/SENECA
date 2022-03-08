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
