#!/bin/sh

# Script to prepare a target folder to be copied to the evaluation board
#
# The folder will contain a python script to run the compiled model on the FPGA,
# a python script to evaluate performances, and two folders containing input and labels images

python prepare_board_images.py -s "$1" -nim "$2"

cp application/app_mt.py build/target/
cp application/scores.py build/target/

if [ "$3" = zcu102 ]; then
    dir=build/compiled_zcu102/unet.xmodel
elif [ "$3" = zcu104 ]; then
    dir=build/compiled_zcu104/unet.xmodel
elif [ "$3" = vck190 ]; then
    dir=build/compiled_vck190/unet.xmodel
elif [ "$3" = u50 ]; then
    dir=build/compiled_u50/unet.xmodel
else
      echo  "Target not found. Valid choices are: zcu102, zcu104, vck190, u50 ..exiting"
      exit 1
fi

cp $dir build/target/
