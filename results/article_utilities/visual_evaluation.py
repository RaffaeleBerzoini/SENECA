from tensorflow.keras.models import load_model
from scores_losses import foc_tversky_loss, dice, dice_liver, dice_bladder, dice_lungs, \
    dice_kidneys, dice_bones

from masks_evaluation import prepare_prediction
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = load_model('../1M-f_model.h5', custom_objects={'foc_tversky_loss': foc_tversky_loss, 'dice': dice,
                                                       'dice_liver': dice_liver,
                                                       'dice_bladder': dice_bladder,
                                                       'dice_lungs': dice_lungs,
                                                       'dice_kidneys': dice_kidneys,
                                                       'dice_bones': dice_bones})
colormap = 'gray'

plt.close("all")

# slice = '86-456' # reni + fegato + spina dorsale
# slice = '127-761' # polmoni + costole
slice = '21-123' # vescica

filename = 'bladder'

original = np.load('../../build/target/images/' + slice + '.npy')
plt.imshow(original, cmap=colormap)
plt.axis('off')
plt.savefig(filename + '-input.eps', format='eps')
plt.show()

prediction_fpga = np.load('predictions/pred_' + slice + '.npy')
plt.imshow(prepare_prediction(prediction_fpga) * 51, cmap=colormap)
plt.axis('off')
plt.savefig(filename + '-fpga.eps', format='eps')
plt.show()

x = np.zeros((8,) + (256,256) + (1,), dtype="float32")
for j in range(8):
    x[j] = np.expand_dims(original, 2)
prediction_gpu = model.predict(x)
plt.imshow(prepare_prediction(prediction_gpu[0]) * 51, cmap=colormap)
plt.axis('off')
plt.savefig(filename + '-gpu.eps', format='eps')
plt.show()

label = cv2.imread('../../build/target/labels/' + slice + '.bmp', 0)
plt.imshow(label * 51, cmap=colormap)
plt.axis('off')
plt.savefig(filename + '-label.eps', format='eps')
plt.show()