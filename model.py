from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

DIVIDER = '-----------------------------------------'


def cba(unet, num_filters=16, kernel_size=(3, 3), activation='relu', padding='same'):
    """
    Convolution - BatchNorm - Activation
    """
    unet = layers.Conv2D(num_filters, kernel_size, padding=padding)(unet)
    unet = layers.BatchNormalization()(unet)
    unet = layers.Activation(activation)(unet)

    return unet


def down_block(unet, num_filters, kernel_size=(3, 3), activation='relu', padding='same', dropout_rate=0.05,
               pool_size=(2, 2), pool_and_drop=True, concatenations=None):
    """
    Convolution - BatchNorm - Activation - Max Pooling - Dropout
    """
    if concatenations is None:
        concatenations = []
    unet = cba(unet, num_filters, kernel_size, activation, padding)
    unet = cba(unet, num_filters, kernel_size, activation, padding)

    concatenations.append(unet)

    if pool_and_drop is True:
        unet = layers.MaxPooling2D(pool_size=pool_size)(unet)
        unet = layers.Dropout(rate=dropout_rate)(unet)

    return unet


def up_block(unet, num_filters, concatenations, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same',
             dropout_rate=0.05):
    unet = layers.Conv2DTranspose(num_filters, kernel_size, strides=strides, padding=padding)(unet)
    unet = layers.concatenate([unet, concatenations.pop()])
    unet = layers.Dropout(rate=dropout_rate)(unet)
    unet = cba(unet, num_filters, kernel_size, activation, padding)
    unet = cba(unet, num_filters, kernel_size, activation, padding)

    return unet


def get_model(img_size, num_classes, batch_size, num_layers=4, num_filters=16):
    inputs = keras.Input(shape=img_size + (1,), batch_size=batch_size)  # (img_h, img_w, 1)
    unet = inputs
    LAYERS = num_layers

    concatenations = []

    for i in range(LAYERS):
        unet = down_block(unet, num_filters=num_filters * (2 ** i),
                          concatenations=concatenations)  # up to 16*32 filters

    unet = down_block(unet, num_filters=num_filters * (2 ** LAYERS), pool_and_drop=False, concatenations=concatenations)

    concatenations.pop()  # remove last element, not needed for upscaling

    for i in range(LAYERS):
        unet = up_block(unet, num_filters=num_filters * (2 ** (LAYERS - i)), concatenations=concatenations)

    output = layers.Conv2D(num_classes, (3, 3), padding='same', activation='softmax')(unet)

    # define and return the model
    return Model(inputs, output)


if __name__ == '__main__':
    im_size = (256, 256)
    n_classes = 6
    batch_dim = 8
    model = get_model(img_size=im_size, num_classes=n_classes, batch_size=batch_dim, num_layers=4, num_filters=12)

    print('\n' + DIVIDER)
    print(' Model Summary')
    print(DIVIDER)
    print(model.summary())
    print("Model Inputs: {ips}".format(ips=model.inputs))
    print("Model Outputs: {ops}".format(ops=model.outputs))
