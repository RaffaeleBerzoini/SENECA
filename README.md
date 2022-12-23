# SENECA: Push Efficient Medical Semantic Segmentation to the Edge


## Setup 

* Open a command prompt and execute:
    ```console
    git clone https://github.com/Xilinx/Vitis-AI.git
    cd Vitis-AI
    git checkout 1.4.1
    ```
* Follow the Vitis-AI installation process [here](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Setting-Up-the-Host) 
  * Once the installation is completed open a terminal in the Vitis-AI directory and execute:  
  ```console
  git clone https://github.com/RaffaeleBerzoini/SENECA.git
  ./docker_run.sh xilinx/vitis-ai-gpu:latest
  ```

The working directory should look similar to:

```text
SENECA   # your WRK_DIR
.
├── application
├── build
├── charts
├── preprocessing
    ├── extract_slices.py
    └── prepare_dataset.sh
├── results
├── ...
└── .py files
```

* Download the [dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890)
  * Data wil be downloaded in a folder named _OrganSegmentations_. If not rename it
  * Move the _OrganSegmentations_ folder in `WRK_DIR/preprocessing`. Now the workspace should look like:

```text
SENECA   # your WRK_DIR
.
├── application
├── preprocessing
    ├── OrganSegmentations
        ├── labels-0.nii.gz
        ├── ...
        └── volume-139.nii.gz
    ├── extract_slices.py
    └── prepare_dataset.sh
├── ...
└── .py files
```

* In the command prompt execute:
  ```console
    Vitis-AI /workspace > conda activate vitis-ai-tensorflow2
    (vitis-ai-tensorflow2) Vitis-AI /workspace > cd SENECA
    (vitis-ai-tensorflow2) Vitis-AI /workspace/SENECA > pip install -r requirements.txt
    (vitis-ai-tensorflow2) Vitis-AI /workspace/SENECA > cd preprocessing
    (vitis-ai-tensorflow2) Vitis-AI /workspace/SENECA/preprocessing > sh prepare_dataset_sh
    (vitis-ai-tensorflow2) Vitis-AI /workspace/SENECA/preprocessing > cd ..
    ```
* Wait for the slice extraction. This could take several minutes

## Training

Now you should be in the WRK_DIR with the following setup:

```text
SENECA   # your WRK_DIR
.
├── ...
├── build
    ├── dataset
        ├── input
        └── target
├── ...
└── .py files
```

In the WRK_DIR execute:

  ```console
python train.py --batchsize 8 --layers 4 --filters 8 --epochs 75
  ```

to train the 1 million parameters model. To test other configurations reported on the paper follow this table:

| **Configuration** | **--layers** | **--filters** | **Parameters [x 10<sup>6</sup>]** |
|:-----------------:|:----------:|:-----------:|:-----------------------:|
|         1M        |      4     |      8      |         ~ 1.034         |
|         2M        |      5     |      6      |         ~ 2.329         |
|         4M        |      5     |      8      |         ~ 4.136         |
|         8M        |      5     |      11     |         ~ 7.814         |
|        16M        |      5     |      16     |         ~ 16.522        |

During training, each time validation results improve, a float model is saved in:
`build/float_model/{val_loss:.4f}-f_model.h5`

## Quantization

You can perform Post Training Quantization (PTQ) or Fast Finetuning Quantization (FFQ) to quantize the float model. 
PTQ is to be preferred in terms of time and computation needs. Try FFQ if you're experiencing performance losses after PTQ

### 1. Post Training Quantization

In the WRK_DIR execute:
  ```console
python quantize.py -m build/float_model/0.1021-f_model.h5 --evaluate --calibration 500
  ```
* You would want to try different calibration dimensions if there is a lot of performance loss after quantization

### 2. Fast Finetuning Quantization

In the WRK_DIR execute:

```console
python quantize.py -m build/float_model/0.1021-f_model.h5 --evaluate --calibration 100 --fastfinetuning --fftepochs 5
```

* Modify the `fast_ft_epochs` as you like
* Keep in mind that FFT requires more memory as you increase the calibration dataset dimensions and the number of FFT epochs


Note that here `0.1021-f_model.h5` is just an example. Check in your `build/float_model/` directory which float models have been generated during training.

* The quantized model is saved in `build/quant_model/q_model.h5`.

## Compilation

To compile the `q_model.h5` for the FPGA execute one of these command.
* `sh compile.sh ZCU102` for the ZCU102
* `sh compile.sh ZCU104` for the ZCU104
* `sh compile.sh vck190` for the VCK190 
* `sh compile.sh u50` for the ALVEO U50

For the ZCU104 (used for this work) the compiled model is saved in `build/compiled_zcu104/` directory

## Deployment on the evaluation board

Set up the evaluation board (we used the ZCU104 for this work) as stated [here](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Setting-Up-the-Evaluation-Board).


In the WRK_DIR execute:

```shell
sh deployment_setup.sh 0 100 zcu104
```

* The firsts two args stands for
  1. Starting image in the images directories list
  2. Number of images to be prepared 
  
* Change the third arg (`zcu104` in our case) if your target board is different

Copy the `build/target/` directory to your board with `scp -r build/target/ root@root@192.168.1.227:~/.` assuming that the target board IP address is 192.168.1.227 - adjust this as appropriate for your system.

You could also directly copy the folder to the board SD card 

On the board execute:
```shell
root@xilinx-zcu104-2021_1:~# cd target
root@xilinx-zcu104-2021_1:~/target# python3 app_mt.py --threads 4 --model unet.xmodel --save
Command line options:
 --image_dir :  images
 --threads   :  4
 --model     :  unet.xmodel
 --save      :  True
------------------------------------
Pre-processing 100 images...
Starting 4 threads...
------------------------------------
Throughput=274.73 fps, total frames = 100, time=0.3640 seconds
Saving  100  predictions...
------------------------------------
```

To evaluate results:

```shell
python3 scores.py       
Command line options:
 --image_dir :  predictions
 --label_dir :  labels
------------------------------
------------------------------
Global  dice :
Mean on slices: 88.77 +- 10.02
Weighted Mean on organs: 93.04 +- 0.07
------------------------------
Organs  dice
Liver: 91.63 +- 0.09
Bladder: 79.21 +- 0.09
Lungs: 96.16 +- 0.09
Kidneys: 81.32 +- 0.08
Bones: 94.35 +- 0.03
```

The script prints out also other metrics for a more complete analysis.

<a id="paper_ref"></a>
# Associated Publication

If you find this repository useful, please use the following citation:

```
@inproceedings{berzoini2021onhow,
  title={On How to Push Efficient Medical Semantic Segmentation to the Edge: the SENECA approach},
  author={Berzoini, Raffaele and D'Arnese, Eleonora and Conficconi, Davide},
  booktitle={2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)},
  year={2022},
  organization={IEEE}
}
