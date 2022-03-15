#!/bin/sh

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
