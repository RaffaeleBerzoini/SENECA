import argparse
import tensorflow as tf
import sys
from dataset_utils import get_DataGen
from tensorflow.keras.models import load_model
from scores_losses import foc_tversky_loss, dice, dice_liver, dice_bladder, dice_lungs, \
    dice_kidneys, dice_bones
from masks_evaluation import evaluate_results

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


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', type=str, default='build/float_model/f_model.h5',
                    help='Full path of model to evaluate. Default is build/float_model/f_model.h5')
    ap.add_argument('-b', '--batchsize', type=int, default=16, help='Batchsize for quantization. Default is 16')
    ap.add_argument('-d', '--imgsize', type=int, default=256, help='Dimension for data generator. Default is 256')
    args = ap.parse_args()

    print('\n------------------------------------')
    print('TensorFlow version : ', tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print('Command line options:')
    print(' --model        : ', args.model)
    print(' --batchsize    : ', args.batchsize)
    print(' --imgsize      : ', args.imgsize)
    print('------------------------------------\n')

    model = load_model(args.model, custom_objects={'foc_tversky_loss': foc_tversky_loss, 'dice': dice,
                                                   'dice_liver': dice_liver,
                                                   'dice_bladder': dice_bladder,
                                                   'dice_lungs': dice_lungs,
                                                   'dice_kidneys': dice_kidneys,
                                                   'dice_bones': dice_bones})

    model.evaluate(get_DataGen(train=False, batch_size=args.batchsize, img_size=(args.imgsize, args.imgsize), calibration=True))

    datagen = get_DataGen(train=False, batch_size=args.batchsize, img_size=(args.imgsize, args.imgsize), calibration=True)
    preds = []
    true = []
    for i in range(900//args.batchsize):
        x, y = datagen.__getitem__(i)
        predictions = model.predict(x)
        for j in range(args.batchsize):
            preds.append(predictions[j])
            true.append(y[j])

    print(len(preds))
    print(len(true))

    evaluate_results(preds, true)


if __name__ == "__main__":
    main()
