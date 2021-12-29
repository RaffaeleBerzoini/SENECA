import os
import cv2
import argparse
import numpy as np


def prepare_filepaths():
    target_dir = "build/dataset/Train-target-PNG-1/"
    target_dir2 = "build/dataset/Train-target-PNG-2/"
    target_dir3 = "build/dataset/Train-target-PNG-3/"
    target_dir4 = "build/dataset/Train-target-PNG-4/"
    target_dir5 = "build/dataset/Train-target-PNG-5/"

    input_dir1 = "build/dataset/Train-Input-PNG-1/"
    input_dir2 = "build/dataset/Train-Input-PNG-2/"
    input_dir3 = "build/dataset/Train-Input-PNG-3/"
    input_dir4 = "build/dataset/Train-Input-PNG-4/"
    input_dir5 = "build/dataset/Train-Input-PNG-5/"

    input_img_paths_1 = sorted(
        [
            os.path.join(input_dir1, fname)
            for fname in os.listdir(input_dir1)
            if fname.endswith(".bmp")
        ]
    )

    input_img_paths_2 = sorted(
        [
            os.path.join(input_dir2, fname)
            for fname in os.listdir(input_dir2)
            if fname.endswith(".bmp")
        ]
    )

    input_img_paths_3 = sorted(
        [
            os.path.join(input_dir3, fname)
            for fname in os.listdir(input_dir3)
            if fname.endswith(".bmp")
        ]
    )

    input_img_paths_4 = sorted(
        [
            os.path.join(input_dir4, fname)
            for fname in os.listdir(input_dir4)
            if fname.endswith(".bmp")
        ]
    )

    input_img_paths_5 = sorted(
        [
            os.path.join(input_dir5, fname)
            for fname in os.listdir(input_dir5)
            if fname.endswith(".bmp")
        ]
    )

    input_img_paths = input_img_paths_1 + input_img_paths_2 + input_img_paths_3 + input_img_paths_4 + input_img_paths_5

    target_img_paths_1 = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".bmp")
        ]
    )

    target_img_paths_2 = sorted(
        [
            os.path.join(target_dir2, fname)
            for fname in os.listdir(target_dir2)
            if fname.endswith(".bmp")
        ]
    )

    target_img_paths_3 = sorted(
        [
            os.path.join(target_dir3, fname)
            for fname in os.listdir(target_dir3)
            if fname.endswith(".bmp")
        ]
    )

    target_img_paths_4 = sorted(
        [
            os.path.join(target_dir4, fname)
            for fname in os.listdir(target_dir4)
            if fname.endswith(".bmp")
        ]
    )

    target_img_paths_5 = sorted(
        [
            os.path.join(target_dir5, fname)
            for fname in os.listdir(target_dir5)
            if fname.endswith(".bmp")
        ]
    )

    target_img_paths = target_img_paths_1 + target_img_paths_2 + target_img_paths_3 + target_img_paths_4 + target_img_paths_5

    print("Number of samples:", len(input_img_paths))
    print("Number of samples:", len(target_img_paths))

    return input_img_paths, target_img_paths


def resize(new_dim):
    input_img_paths, target_img_paths = prepare_filepaths()
    for input, target in zip(input_img_paths, target_img_paths):
        img = cv2.imread(input, 0)
        cv2.imwrite(input, cv2.resize(img, dsize=(new_dim, new_dim), interpolation=cv2.INTER_LINEAR))
        lab = cv2.imread(target, 0)
        cv2.imwrite(target, cv2.resize(lab, dsize=(new_dim, new_dim), interpolation=cv2.INTER_LINEAR))
        print(input, "|", target)


def scale():
    input_img_paths, target_img_paths = prepare_filepaths()
    print("Scaling the images between -1 and 1...")
    for input in input_img_paths:
        # print(input)
        img = cv2.imread(input, 0)
        img = img / 127.5
        img = img - 1
        np.save(input[:-3] + "npy", img)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dimension', type=int, default=128,
                    help='New images dimensions. Must be an integer. Default is 128.')
    ap.add_argument('-r', '--resize', action='store_true', help='Resize the images. Default is False.')
    ap.add_argument('-s', '--scale', action='store_true', help='Scale the images. Default is False.')
    args = ap.parse_args()

    if args.resize is True:
        resize(args.dimension)
    if args.scale is True:
        scale()
