import os

import numpy as np
import cv2
from imutils.contours import sort_contours


def cut_lines(name):
    img = cv2.imread("data/test/%s" % name, 0)

    (thresh, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    img_bin = 255 - img_bin
    cv2.imshow("img", img_bin)
    cv2.waitKey(0)

    kernel_length = np.array(img).shape[1] // 80
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    img_temp = cv2.erode(img_bin, hor_kernel, iterations=2)
    hor_lines_img = cv2.dilate(img_temp, hor_kernel, iterations=2)

    cv2.imshow("img", hor_lines_img)
    cv2.waitKey(0)

    img_final_bin = cv2.bitwise_not(hor_lines_img)

    cv2.imshow("img", img_final_bin)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (contours, bboxes) = sort_contours(contours, method="top-to-bottom")

    img_bin = 255 - img_bin

    img_bin = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)

    i = 0

    if not os.path.exists("data/test/lines/%s/" % name.strip(".png")):
        os.mkdir("data/test/lines/%s/" % name.strip(".png"))

    print(img.shape)

    res_contours = []

    for c in contours:
        i += 1
        x, y, w, h = cv2.boundingRect(c)
        if w < img.shape[0] / 3:
            cv2.rectangle(img_bin, (x, y - 22), (x + w, y), (0, 0, 255))
            res_contours.append((x, max(y - 22, 0), w, 22))
            #cv2.imwrite("data/test/lines/%s/%s.png" % (name.strip(".png"), i), img[y: y + h, x: x + w])

    cv2.imshow("img", img_bin)
    cv2.waitKey(0)

    return res_contours


if __name__ == '__main__':
    cut_lines('15.png')
