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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print('gpus true')
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

DIVIDER = '-----------------------------------------'


def train(learnrate, epochs, batch_size, chkpt_dir, tboard, logdir, old_model, starting_epoch):
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

    if old_model is not None:
        print('Loading old model...')
        model = load_model(old_model, custom_objects={'foc_tversky_loss': foc_tversky_loss, 'dice': dice,
                                                      'dice_liver': dice_liver,
                                                      'dice_bladder': dice_bladder,
                                                      'dice_lungs': dice_lungs,
                                                      'dice_kidneys': dice_kidneys,
                                                      'dice_bones': dice_bones})
    else:
        model = get_model(img_size=img_size, num_classes=num_classes, batch_size=batch_size, num_layers=4,
                          num_filters=8)
        model.compile(optimizer=Adam(learning_rate=learnrate),
                      loss=foc_tversky_loss,
                      metrics=[dice, dice_liver, dice_bladder, dice_lungs, dice_kidneys,
                               dice_bones])

    # If something wrong with next training i've moved .compile before .summary

    print('\n' + DIVIDER)
    print(' Model Summary')
    print(DIVIDER)
    print(model.summary())
    print("Model Inputs: {ips}".format(ips=model.inputs))
    print("Model Outputs: {ops}".format(ops=model.outputs))

    '''
    tf.data pipelines
    '''
    # train and test folder
    train_dataset = get_DataGen(train=True, batch_size=batch_size, img_size=img_size)
    val_dataset = get_DataGen(train=False, batch_size=batch_size, img_size=img_size)

    '''
    Call backs
    '''
    tb_call = TensorBoard(log_dir=tboard)

    chkpt_call = ModelCheckpoint(
        filepath=os.path.join(chkpt_dir, '{val_loss:.4f}-f_model.h5'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True)

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
    train_history = model.fit(train_dataset,
                              epochs=epochs,
                              validation_data=val_dataset,
                              callbacks=callbacks_list,
                              verbose=1)

    print(
        "\nTensorBoard can be opened with the command: tensorboard --logdir={dir} --host localhost --port 6006".format(
            dir=tboard))

    return


def run_main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--batchsize', type=int, default=64,
                    help='Training batchsize. Must be an integer. Default is 64.')
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
                    help='Number of epoch we are starting from to keep consistent learning rate and epoch saving. Default is 0')
    args = ap.parse_args()

    print('\n' + DIVIDER)
    print('Keras version      : ', tf.keras.__version__)
    print('TensorFlow version : ', tf.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print('--batchsize    : ', args.batchsize)
    print('--learnrate    : ', args.learnrate)
    print('--epochs       : ', args.epochs)
    print('--chkpt_dir    : ', args.chkpt_dir)
    print('--tboard       : ', args.tboard)
    print('--logdir       : ', args.logdir)
    print('--model        : ', args.model)
    print('--start_epoch  : ', args.start_epoch)
    print(DIVIDER)

    train(args.learnrate, args.epochs, args.batchsize, args.chkpt_dir, args.tboard, args.logdir, args.model,
          args.start_epoch)


if __name__ == '__main__':
    run_main()
