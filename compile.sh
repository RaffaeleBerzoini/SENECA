#!/bin/bash

set -e

if [ "$#" -eq 2 ]; then
	BOARD=$1
	MODEL_NAME=$2
else
	echo "Error: please provide BOARD and MODEL_NAME as arguments."
	echo "Example: ./compile.sh KV260 path/to/quant_model.h5"
	exit 1
fi

sudo mkdir -p /opt/vitis_ai/compiler/arch/DPUCZDX8G/Ultra96/
sudo cp arch.json /opt/vitis_ai/compiler/arch/DPUCZDX8G/Ultra96/

compile() {
      vai_c_tensorflow2 \
            --model           ${MODEL_NAME} \
            --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/${BOARD}/arch.json \
		        --output_dir ./xmodel \
		        --net_name seneca4ctorg${BOARD}${MODEL}
}

mkdir -p xmodel/
compile 2>&1 | tee build/logs/compile_$TARGET.log


echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"
