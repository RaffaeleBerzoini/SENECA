#!/bin/bash

set -e

if [ "$#" -eq 2 ]; then
	BOARD=$1
	MODEL_NAME=$2
else
	echo "Error: please provide BOARD and MODEL_NAME as arguments."
	echo "Example: ./compile.sh KV260 cf_resnet50_imagenet_224_224_7.7G_1.4"
	exit 1
fi

compile() {
      vai_c_tensorflow2 \
            --model           ${MODEL_NAME} \
            --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/${BOARD}/arch.json \
		        --output_dir . \
		        --net_name tf2_${MODEL}
}


compile 2>&1 | tee build/logs/compile_$TARGET.log


echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"