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

import argparse

import numpy as np
import os
import cv2

n_liver = 0
n_bladder = 0
n_lungs = 0
n_kidneys = 0
n_bones = 0

divider = '------------------------------'


def explode_img(img, num_classes=6):
    exploded = np.zeros(shape=img.shape + (num_classes,), dtype=np.uint8)
    for i in range(num_classes):
        exploded[:, :, i] = (img[:, :] == i).astype(np.uint8)
    return exploded


def prepare_prediction(pred):
    mask = np.argmax(pred, axis=-1)
    mask = mask.astype('uint8')
    return mask


def sensitivity_single(pred, true):
    TP = np.sum(pred * true)
    FN = np.sum(true * (1 - pred))
    return (TP + 1) / (TP + FN + 1)


def specificity_single(pred, true):
    TN = np.sum((1 - true) * (1 - pred))
    FP = np.sum((1 - true) * pred)
    return (TN + 1) / (TN + FP + 1)


def dice_single(pred, true):
    intersection = np.sum(pred * true)
    union = np.sum(pred + true)
    return (2 * intersection + 1) / (union + 1)


def score_total(pred, true, score):
    true = explode_img(true)
    score_background = score(pred == 0, true[:, :, 0])
    score_liver = score(pred == 1, true[:, :, 1])
    score_bladder = score(pred == 2, true[:, :, 2])
    score_lungs = score(pred == 3, true[:, :, 3])
    score_kidneys = score(pred == 4, true[:, :, 4])
    score_bones = score(pred == 5, true[:, :, 5])

    background_w = np.sum(true[:, :, 0])
    liver_w = np.sum(true[:, :, 1])
    bladder_w = np.sum(true[:, :, 2])
    lungs_w = np.sum(true[:, :, 3])
    kidneys_w = np.sum(true[:, :, 4])
    bones_w = np.sum(true[:, :, 5])

    global n_liver
    n_liver += liver_w
    global n_bladder
    n_bladder += bladder_w
    global n_lungs
    n_lungs += lungs_w
    global n_kidneys
    n_kidneys += kidneys_w
    global n_bones
    n_bones += bones_w
    # print((str(score_liver) + ', ')*(int(liver_w//250)), end=' ')

    return ((liver_w * score_liver + bladder_w * score_bladder + lungs_w * score_lungs +
             kidneys_w * score_kidneys + bones_w * score_bones + 1) / (
                    liver_w + bladder_w + lungs_w + kidneys_w + bones_w + 1)), score_liver * liver_w, score_bladder * bladder_w, score_lungs * lungs_w, score_kidneys * kidneys_w, score_bones * bones_w


def evaluate_results(dir_pred='predictions', dir_true='labels'):
    list_pred = os.listdir(dir_pred)
    list_true = os.listdir(dir_true)

    list_pred = sorted(list_pred)
    list_true = sorted(list_true)

    for pred, true in zip(list_pred, list_true):
        assert pred[5:-4] == true[:-4]

    metrics = [dice_single, sensitivity_single, specificity_single]
    metrics_labels = ['dice', 'sensitivity', 'specificity']

    for metric, metrics_label in zip(metrics, metrics_labels):
        global n_liver
        global n_bladder
        global n_lungs
        global n_kidneys
        global n_bones

        n_liver = 0
        n_bladder = 0
        n_lungs = 0
        n_kidneys = 0
        n_bones = 0

        scores = []
        score = 0
        current_score = 0
        score_liver_total = []
        score_bladder_total = []
        score_lungs_total = []
        score_kidneys_total = []
        score_bones_total = []
        for i in range(len(list_pred)):
            pred = np.load(dir_pred + '/' + list_pred[i])
            true = cv2.imread(dir_true + '/' + list_true[i], 0)
            pred = prepare_prediction(pred)
            current_score, score_liver, score_bladder, score_lungs, score_kidneys, score_bones = score_total(pred, true,
                                                                                                             metric)

            score_liver_total.append(score_liver)
            score_bladder_total.append(score_bladder)
            score_lungs_total.append(score_lungs)
            score_kidneys_total.append(score_kidneys)
            score_bones_total.append(score_bones)
            score += current_score
            scores.append(current_score)

            scores.append(current_score)

        print(divider)
        print('Global ', metrics_label, ':')
        print("Mean on slices: %.2f +- %.2f" % (np.mean(scores) * 100, np.std(scores) * 100))

        std_organs = ((np.std(score_liver_total) + np.std(score_bladder_total) + np.std(score_lungs_total) +
                       np.std(score_kidneys_total) + np.std(score_bones_total)) / (n_liver + n_bladder + n_lungs +
                                                                                   n_kidneys + n_bones))
        print('Weighted Mean on organs: %.2f +- %.2f' % (
            (np.sum(score_liver_total) + np.sum(score_bladder_total) + np.sum(score_lungs_total) + np.sum(
                score_kidneys_total)
             + np.sum(score_bones_total)) / (n_liver + n_bladder + n_lungs + n_kidneys + n_bones) * 100,
            std_organs * 100))
        print(divider)

        print('Organs ', metrics_label)
        print('Liver: %.2f +- %.2f' % (
            np.sum(score_liver_total) / n_liver * 100, np.std(score_liver_total) / n_liver * 100))
        print('Bladder: %.2f +- %.2f' % (
            np.sum(score_bladder_total) / n_bladder * 100, np.std(score_bladder_total) / n_bladder * 100))
        print('Lungs: %.2f +- %.2f' % (
            np.sum(score_lungs_total) / n_lungs * 100, np.std(score_lungs_total) / n_lungs * 100))
        print('Kidneys: %.2f +- %.2f' % (
            np.sum(score_kidneys_total) / n_kidneys * 100, np.std(score_kidneys_total) / n_kidneys * 100))
        print('Bones: %.2f +- %.2f' % (
            np.sum(score_bones_total) / n_bones * 100, np.std(score_bones_total) / n_bones * 100))
        print(divider)


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-id', '--image_dir', type=str, default='predictions',
                    help='Path to folder of images. Default is predictions')
    ap.add_argument('-ld', '--label_dir', type=str, default='labels',
                    help='Path to folder of labels. Default is labels')
    args = ap.parse_args()

    print('Command line options:')
    print(' --image_dir : ', args.image_dir)
    print(' --label_dir : ', args.label_dir)
    print(divider)
    evaluate_results(args.image_dir, args.label_dir)


if __name__ == '__main__':
    main()
