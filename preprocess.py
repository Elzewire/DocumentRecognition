import os

import cv2
import numpy
from PIL import Image, ImageStat


# Preparing image
def prepare(name):
    image = Image.open(name)
    image = image.convert('L')

    channels = image.split()
    stat = ImageStat.Stat(image)

    # Threshing image
    threshold = stat.mean[0] * 0.75
    threshed = channels[0].point(lambda p: p > threshold and 255)
    image = Image.merge('RGB', (threshed, threshed, threshed))

    imcv = numpy.array(image)
    imcv_gray = cv2.cvtColor(imcv, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(imcv_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    noise = [1, 9, 12, 16, 15]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        squares.append(w * h)
        if w * h > 25:
            cv2.rectangle(imcv, (x, y), (x + w, y + h), (0, 0, 255), 1)

    cv2.imshow("img", imcv)
    cv2.waitKey(0)

    d = {}
    for s in squares:
        if s in d.keys():
            d[s] += 1
        else:
            d[s] = 1

    d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    print(d)
    return imcv


if __name__ == '__main__':
    prepare('data/test/15.png')
