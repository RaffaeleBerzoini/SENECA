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


import os
import sys
import argparse

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import load_model

from scores_losses import foc_tversky_loss, dice, dice_liver, \
    dice_bladder, dice_lungs, dice_kidneys, dice_bones
from model import get_model
from dataset_utils import get_DataGen
from GPU_MEMORY import MAX_MEM

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print('gpus true')
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MAX_MEM)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

DIVIDER = '-----------------------------------------'


def train(learnrate, epochs, batch_size, layers, filters, chkpt_dir, tboard, logdir, old_model, starting_epoch):
    def step_decay(epoch):
        """
        Learning rate scheduler used by callback
        Reduces learning rate depending on number of epochs
        """
        epoch += starting_epoch
        lr = learnrate
        if epoch > 68:
            lr = 0.000001
        elif epoch > 58:
            lr = 0.000005
        elif epoch > 48:
            lr = 0.00001
        elif epoch > 36:
            lr = 0.00005
        elif epoch > 24:
            lr = 0.0001
        elif epoch > 12:
            lr = 0.0005

        return lr

    '''
    Define the model
    '''

    img_size = (256, 256)
    num_classes = 6

    """Loads float-model to continue training"""
    if old_model is not None:
        print('Loading old model...')
        model = load_model(old_model, custom_objects={'foc_tversky_loss': foc_tversky_loss, 'dice': dice,
                                                      'dice_liver': dice_liver,
                                                      'dice_bladder': dice_bladder,
                                                      'dice_lungs': dice_lungs,
                                                      'dice_kidneys': dice_kidneys,
                                                      'dice_bones': dice_bones})
    else:
        model = get_model(img_size=img_size, num_classes=num_classes, batch_size=batch_size, num_layers=layers,
                          num_filters=filters)
        model.compile(optimizer=Adam(learning_rate=learnrate),
                      loss=foc_tversky_loss,
                      metrics=[dice, dice_liver, dice_bladder, dice_lungs, dice_kidneys,
                               dice_bones])

    print('\n' + DIVIDER)
    print(' Model Summary')
    print(DIVIDER)
    print(model.summary())
    print("Model Inputs: {ips}".format(ips=model.inputs))
    print("Model Outputs: {ops}".format(ops=model.outputs))

    '''
    tf.data pipelines
    '''
    # train and validation batch-generators
    train_dataset = get_DataGen(dataset="train", batch_size=batch_size, img_size=img_size)
    val_dataset = get_DataGen(dataset="test", batch_size=batch_size, img_size=img_size)

    '''
    Call backs
    '''
    tb_call = TensorBoard(log_dir=tboard)

    # float-model saving during training
    chkpt_call = ModelCheckpoint(
        filepath=os.path.join(chkpt_dir, '{val_loss:.4f}-f_model.h5'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True)

    # learning-rate changes as the number of epochs increases
    lr_scheduler_call = LearningRateScheduler(schedule=step_decay,
                                              verbose=1)

    csv_logger = CSVLogger(logdir, append=True, separator=';')

    callbacks_list = [tb_call, chkpt_call, lr_scheduler_call, csv_logger]

    '''
    Training
    '''

    print('\n' + DIVIDER)
    print(' Training model with training set..')
    print(DIVIDER)

    # make folder for saving trained model checkpoint
    os.makedirs(chkpt_dir, exist_ok=True)  # does not raise an error if the directory already exists

    # run training
    model.fit(train_dataset,
              epochs=epochs,
              validation_data=val_dataset,
              validation_freq=3,
              callbacks=callbacks_list,
              verbose=1)

    print(
        "\nTensorBoard can be opened with the command: tensorboard --logdir={dir} --host localhost --port 6006".format(
            dir=tboard))

    return


def run_main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--batchsize', type=int, default=8,
                    help='Training batchsize. Must be an integer. Default is 8.')
    ap.add_argument('-l', '--layers', type=int, default=4,
                    help='Number of layers for the U-Net. Number considers only the encoding path. Must be an '
                         'integer. Default is 4')
    ap.add_argument('-f', '--filters', type=int, default=8,
                    help='Number of filters for the U-Net. Must be an integer. Default is 8')
    ap.add_argument('-e', '--epochs', type=int, default=5,
                    help='number of training epochs. Must be an integer. Default is 5.')
    ap.add_argument('-lr', '--learnrate', type=float, default=0.001,
                    help='optimizer learning rate. Must be floating-point value. Default is 0.001')
    ap.add_argument('-cf', '--chkpt_dir', type=str, default='build/float_model',
                    help='Path and name of folder for storing Keras checkpoints. Default is build/float_model')
    ap.add_argument('-tb', '--tboard', type=str, default='build/tb_logs',
                    help='path to folder for saving TensorBoard data. Default is build/tb_logs.')
    ap.add_argument('-log', '--logdir', type=str, default='train_log.csv',
                    help='path to csv file to save training log. Default is train_log.csv.')
    ap.add_argument('-m', '--model', type=str, default=None,
                    help='Model to load to continue the training. Default is None')
    ap.add_argument('-se', '--start_epoch', type=int, default=0,
                    help='Number of epoch we are starting from to keep consistent learning rate and epoch saving. '
                         'Default is 0')
    args = ap.parse_args()

    print('\n' + DIVIDER)
    print('Keras version      : ', tf.keras.__version__)
    print('TensorFlow version : ', tf.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print('--batchsize    : ', args.batchsize)
    print('--layers       : ', args.layers)
    print('--filters      : ', args.filters)
    print('--learnrate    : ', args.learnrate)
    print('--epochs       : ', args.epochs)
    print('--chkpt_dir    : ', args.chkpt_dir)
    print('--tboard       : ', args.tboard)
    print('--logdir       : ', args.logdir)
    print('--model        : ', args.model)
    print('--start_epoch  : ', args.start_epoch)
    print(DIVIDER)

    train(args.learnrate, args.epochs, args.batchsize, args.layers, args.filters, args.chkpt_dir, args.tboard,
          args.logdir, args.model, args.start_epoch)


if __name__ == '__main__':
    run_main()
