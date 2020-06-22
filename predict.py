import os

import numpy as np
import cv2
from imutils.contours import sort_contours
from tensorflow.python import keras

from lines import cut_lines


def predict(img_bin, i):
    img_bin = 255 - img_bin
    # cv2.imshow("img", img_bin)
    # cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        (contours, bboxes) = sort_contours(contours, method="top-to-bottom")

    images = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 100:
            res = cv2.resize(img_bin[y: y + h, x: x + w], (28, 28), interpolation=cv2.INTER_NEAREST)
            # cv2.imshow("img", res)
            cv2.waitKey(0)
            images.append(res)
            #print(cv2.imwrite('data/dataset/%s.png' % i, res))
            i += 1

    # cv2.imshow("img", img_bin)
    # cv2.waitKey(0)

    res = ''

    if len(images) > 0:
        images = np.array(images)
        images = images / 255.0
        images = np.expand_dims(images, axis=3)

        model = keras.models.load_model('model.h5')
        predictions = model.predict(np.array(images))
        # check if the highest results have more than .5 possibility
        results = np.argmax(predictions, axis=1)
        possibilities = np.max(predictions, axis=1)

        n = 0
        for im in images:
            # cv2.imshow("result: %s" % results[n], im)
            # cv2.waitKey(0)
            n += 1

        for i in range(results.shape[0]):
            #if possibilities[i] > .4:
            res += results[i].__str__()

    return res, i


def fill_predictions(file, contours):
    img = cv2.imread("data/test/%s" % file, 0)

    # img = cv2.GaussianBlur(img, (2, 2), cv2.BORDER_DEFAULT)

    (thresh, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    i = 0

    for c in contours:
        x, y, w, h = c
        #cv2.imshow("img", img[y: y + h, x: x + w])
        #cv2.waitKey(0)
        result, i = predict(img_bin[y: y + h, x: x + w], i)
        cv2.rectangle(img_bin, (x, y), (x + w, y + h), (255, 255, 255), -1)
        t_size = cv2.getTextSize(result, cv2.FONT_HERSHEY_TRIPLEX, .5, 1)[0]
        tX, tY = (x + w // 2 - t_size[0] // 2), (y + h // 2 + t_size[1] // 2)
        img_bin = cv2.putText(img_bin, result, (tX, tY), cv2.FONT_HERSHEY_TRIPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA,
                              False)
        cv2.imshow("img", img_bin)
        cv2.waitKey(0)


if __name__ == '__main__':

    path = 'data/test/'

    contours = cut_lines('15.png')
    fill_predictions('15.png', contours)

    '''
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            contours = cut_lines(file)
            fill_predictions(file, contours)
    '''

