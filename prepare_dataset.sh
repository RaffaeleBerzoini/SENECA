#!/bin/sh

echo "Making direcotries for training and to be saved files"
mkdir build
mkdir build/dataset

echo "Deleting dataset directory"
rm -R build/dataset

unzip Tars-ng.zip -d build/dataset
mv build/dataset/Tars-ng/* build/dataset
rm -R build/dataset/Tars-ng/

cd build/dataset/
for f in *.tar.gz; do tar xf "$f"; done
rm *.tar.gz

cd ../../

matlab -batch "imadjusting"

python preprocessing.py --scale
