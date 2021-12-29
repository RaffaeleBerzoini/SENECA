import argparse
import os
import shutil
import sys

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from scores_losses import foc_tversky_loss, foc_tversky_loss_2, dice, dice_loss, dice_liver, dice_bladder, dice_lungs, \
    dice_kidneys, dice_bones
from dataset_utils import get_DataGen

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print('gpus true')
    # Restrict TensorFlow to only allocate 1GB * 2 of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4500)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

DIVIDER = '-----------------------------------------'


def evaluate_model(model, dataset, quantized=False):
    """
    Evaluate quantized model
    """
    print('\n' + DIVIDER)
    if quantized is True:
        print('Evaluating quantized model...')
    else:
        print('Evaluating float model...')
    print(DIVIDER + '\n')

    test_dataset = dataset

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=foc_tversky_loss,
                  metrics=[dice, dice_liver, dice_bladder, dice_lungs, dice_kidneys,
                           dice_bones])

    scores = model.evaluate(test_dataset)

    if quantized is True:
        print('Quantized model dice score: ')
    else:
        print('Float model dice score: ')
    for score in scores:
        print('{0:.4f}'.format(score * 100), '%')
    print('\n' + DIVIDER)


def quant_model(float_model, quant_model, batchsize, imgsize, evaluate):
    '''
    Quantize the floating-point model
    Save to HDF5 file
    '''
    print(float_model)
    # make folder for saving quantized model
    head_tail = os.path.split(quant_model)
    os.makedirs(head_tail[0], exist_ok=True)

    # load the floating point trained model
    float_model = load_model(float_model, custom_objects={'foc_tversky_loss': foc_tversky_loss, 'dice': dice,
                                                          'dice_liver': dice_liver,
                                                          'dice_bladder': dice_bladder,
                                                          'dice_lungs': dice_lungs,
                                                          'dice_kidneys': dice_kidneys,
                                                          'dice_bones': dice_bones})

    imgsize = (imgsize, imgsize)

    # get input dimensions of the floating-point model
    height = float_model.input_shape[1]
    width = float_model.input_shape[2]

    # Instance of the dataset via keras.utils.Sequence
    quant_dataset = get_DataGen(train=False, batch_size=batchsize, calibration=True)

    # run quantization
    quantizer = vitis_quantize.VitisQuantizer(float_model)

    quantized_model = quantizer.quantize_model(calib_dataset=quant_dataset, verbose=1)
    # quantized_model = quantizer.quantize_model(calib_dataset=quant_dataset, verbose=1, include_fast_ft=True, fast_ft_epochs=10)

    # saved quantized model
    quantized_model.save(quant_model)

    if evaluate:
        evaluate_model(quantized_model,
                       get_DataGen(train=False, batch_size=batchsize, img_size=imgsize, calibration=False),
                       quantized=True)

    return


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--float_model', type=str, default='build/float_model/f_model.h5',
                    help='Full path of floating-point model. Default is build/float_model/k_model.h5')
    ap.add_argument('-q', '--quant_model', type=str, default='build/quant_model/q_model.h5',
                    help='Full path of quantized model. Default is build/quant_model/q_model.h5')
    ap.add_argument('-b', '--batchsize', type=int, default=16, help='Batchsize for quantization. Default is 16')
    ap.add_argument('-d', '--imgsize', type=int, default=256, help='Dimension for data generator. Default is 256')
    ap.add_argument('-e', '--evaluate', action='store_true',
                    help='Evaluate floating-point model if set. Default is no evaluation.')
    args = ap.parse_args()

    print('\n------------------------------------')
    print('TensorFlow version : ', tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print('Command line options:')
    print(' --float_model  : ', args.float_model)
    print(' --quant_model  : ', args.quant_model)
    print(' --batchsize    : ', args.batchsize)
    print(' --imgsize      : ', args.imgsize)
    print(' --evaluate     : ', args.evaluate)
    print('------------------------------------\n')

    quant_model(args.float_model, args.quant_model, args.batchsize, args.imgsize, args.evaluate)


if __name__ == "__main__":
    main()
