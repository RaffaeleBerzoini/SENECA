import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from pynput import keyboard
import cv2
import time


def load_volume_and_segmentation(path):
    vol = nib.load(path).get_fdata()
    seg = np.zeros(shape=(vol.shape[-1],))

    return vol, seg


def segment_volume(path):
    vol, seg = load_volume_and_segmentation(path)
    print('d for next, a for previous:')
    z = 0
    pressed = 0
    with keyboard.Events() as events:
        # Block for as much as possible

        while True:
            time.sleep(1)
            event = events.get(1e6)
            if event.key != keyboard.KeyCode.from_char('q'):
                pressed += 1
                if pressed % 2 == 1:
                    if event.key == keyboard.KeyCode.from_char('d') and z < vol.shape[-1] - 1:
                        z += 1
                        plt.close()
                        plt.imshow(vol[:, :, z])
                        plt.show()
                    if event.key == keyboard.KeyCode.from_char('a') and z > 0:
                        z -= 1
                        plt.close()
                        plt.imshow(vol[:, :, z])
                        plt.show()
                    print(event.key)
            else:
                break

            # if event.key == keyboard.KeyCode.from_char('s'):
            #     print("YES")


segment_volume('OrganSegmentations/volume-0.nii.gz')
