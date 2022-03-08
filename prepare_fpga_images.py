import argparse

from dataset_utils import prepare_target_images

ap = argparse.ArgumentParser()
ap.add_argument('-s', '--start', type=int, default=0, help='Starting image in the images list')
ap.add_argument('-nim', '--numimages', type=int, default=1000, help='Number of images to be prepared for the FPGA')

args = ap.parse_args()

prepare_target_images(args.start, args.numimages)
