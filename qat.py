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
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler

from scores_losses import foc_tversky_loss, dice, dice_loss, dice_liver, dice_bladder, dice_lungs, \
    dice_kidneys, dice_bones
from dataset_utils import get_DataGen

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print('gpus true')
    # Restrict TensorFlow to only allocate 1GB * 2 of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

DIVIDER = '-----------------------------------------'


def evaluate_model(model, dataset, quantized=False):
    '''
    Evaluate quantized model
    '''
    print('\n' + DIVIDER)
    if quantized is True:
        print('Evaluating quantized model...')
    else:
        print('Evaluating float model...')
    print(DIVIDER + '\n')

    test_dataset = dataset

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=dice_loss,
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


def quant_model(float_model, quant_model, batchsize, imgsize, tfrec_dir, evaluate):
    learnrate = 0.001

    def step_decay(epoch):
        """
        Learning rate scheduler used by callback
        Reduces learning rate depending on number of epochs
        """
        lr = learnrate
        if epoch > 47:
            lr = 0.00001
        elif epoch > 35:
            lr = 0.0001

        return lr

    '''
    Quantize the floating-point model
    Save to HDF5 file
    '''
    print(float_model)
    # make folder for saving quantized model
    head_tail = os.path.split(quant_model)
    os.makedirs(head_tail[0], exist_ok=True)

    # load the floating point trained model
    float_model = load_model(float_model, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice,
                                                          'dice_coef_liver': dice_liver,
                                                          'dice_coef_bladder': dice_bladder,
                                                          'dice_coef_lungs': dice_lungs,
                                                          'dice_coef_kidneys': dice_kidneys,
                                                          'dice_coef_bones': dice_bones})

    imgsize = (imgsize, imgsize)

    if (evaluate):
        evaluate_model(float_model, get_DataGen(train=False, batch_size=batchsize, img_size=imgsize, calibration=False))

    # Instance of the dataset via keras.utils.Sequence
    calib_dataset = get_DataGen(train=False, batch_size=batchsize, calibration=True)

    # run quantization
    quantizer = vitis_quantize.VitisQuantizer(float_model, quantize_strategy='8bit_tqt')
    qat_model = quantizer.get_qat_model(
        init_quant=True, calib_dataset=calib_dataset)

    chkpt_call = ModelCheckpoint(filepath=quant_model, monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler_call = LearningRateScheduler(schedule=step_decay,
                                              verbose=1)

    callbacks_list = [chkpt_call, lr_scheduler_call]

    qat_model.compile(optimizer=Adam(learning_rate=learnrate),
                      loss=dice_loss,
                      metrics=[dice, dice_liver, dice_bladder, dice_lungs, dice_kidneys,
                               dice_bones])

    train_dataset = get_DataGen(train=True, batch_size=batchsize, img_size=imgsize)
    val_dataset = get_DataGen(train=False, batch_size=batchsize, img_size=imgsize)

    train_history = qat_model.fit(train_dataset,
                                  epochs=2,
                                  validation_data=val_dataset,
                                  callbacks=callbacks_list,
                                  verbose=1)

    qat_model = load_model(quant_model)  # ricarico per tenere il migliore
    qat_model = vitis_quantize.VitisQuantizer.get_deploy_model(qat_model)
    qat_model.save(qat_model)

    # Load Quantized Model
    quantized_model = load_model(quant_model)
    quantized_model.compile(optimizer=Adam(learning_rate=learnrate),
                            loss=dice_loss,
                            metrics=[dice, dice_liver, dice_bladder, dice_lungs, dice_kidneys,
                                     dice_bones])

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
    ap.add_argument('-q', '--quant_model', type=str, default='build/quant_model_qat/q_model.h5',
                    help='Full path of quantized model. Default is build/quant_model_qat/q_model.h5')
    ap.add_argument('-b', '--batchsize', type=int, default=16, help='Batchsize for quantization. Default is 8')
    ap.add_argument('-d', '--imgsize', type=int, default=256, help='Dimension for data generator. Default is 256')
    ap.add_argument('-tfdir', '--tfrec_dir', type=str, default='build/tfrecords',
                    help='Full path to folder containing TFRecord files. Default is build/tfrecords')
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
    print(' --tfrec_dir    : ', args.tfrec_dir)
    print(' --evaluate     : ', args.evaluate)
    print('------------------------------------\n')

    quant_model(args.float_model, args.quant_model, args.batchsize, args.imgsize, args.tfrec_dir, args.evaluate)


if __name__ == "__main__":
    main()
