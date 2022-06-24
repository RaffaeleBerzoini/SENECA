import nibabel as nib
from tqdm import tqdm
import numpy as np
from nibabel import processing
import matplotlib.pyplot as plt


def adjust_voxel_size(path, vox_size_out, isLabel = False, keep_z_vox_siz=True):
    vol = nib.load(path)

    if keep_z_vox_siz:
        vox_size_out[2] = vol.header['pixdim'][3]

    if isLabel:
        vol_out = processing.conform(vol, out_shape=vol.shape, voxel_size=vox_size_out, order=0)
    else:
        vol_out = processing.conform(vol, out_shape=vol.shape, voxel_size=vox_size_out, order=3)

    return vol_out


def volumes_analysis():
    min_x = np.array((100, 100, 100))
    min_y = np.array((100, 100, 100))
    min_z = np.array((100, 100, 100))
    x_vals = []
    y_vals = []
    z_vals = []

    for i in tqdm(range(140)):
        vol = nib.load(f'OrganSegmentations/volume-{i}.nii.gz')
        # plt.hist(vol.get_fdata().flatten(), bins=100)
        # plt.savefig(f'histograms/hist-{i}.png', dpi=300, bbox_inches='tight')
        # plt.close()

        # plt.show()
        # input("next")
        # print(f'volume-{i}: ', np.max(vol.get_fdata()))
        # print(vol.shape)
        vox_size = vol.header['pixdim'][1:4]
        # print(vox_size)
        min_x = min_x if vox_size[0] > min_x[0] else vox_size
        min_y = min_y if vox_size[1] > min_y[1] else vox_size
        min_z = min_z if vox_size[2] > min_z[2] else vox_size
        x_vals.append(vox_size[0])
        y_vals.append(vox_size[1])
        z_vals.append(vox_size[2])

    print('\nx - min: ', min_x, ', mean: ', np.mean(x_vals))
    print('y - min: ', min_y, ', mean: ', np.mean(y_vals))
    print('z - min: ', min_z, ', mean: ', np.mean(z_vals))

    return np.array([min_x[0], min_y[1], min_z[2]]), np.array([np.mean(x_vals), np.mean(y_vals), np.mean(z_vals)])


if __name__ == '__main__':
    volumes_analysis()
