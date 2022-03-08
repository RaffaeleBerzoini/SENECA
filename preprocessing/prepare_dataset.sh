#!/bin/sh

echo "Deleting existing dataset directory"
rm -R ../build/dataset

mkdir ../build/dataset

mkdir ../build/dataset/input
mkdir ../build/dataset/target

python extract_slices.py