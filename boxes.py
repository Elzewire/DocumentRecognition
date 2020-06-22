import cv2
import os
import numpy as np

from predict import predict
from sort_contours import sort_contours

def cut_boxes(name):
    img = cv2.imread("data/test/%s" % name, 0)

    (thresh, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    img_bin = 255 - img_bin
    cv2.imshow("img", img_bin)
    cv2.waitKey(0)

    kernel_length = np.array(img).shape[1] // 80
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    print(ver_kernel)
    print(hor_kernel)

    img_temp1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    ver_lines_img = cv2.dilate(img_temp1, ver_kernel, iterations=3)

    # cv2.imshow("img", ver_lines_img)
    # cv2.waitKey(0)

    img_temp2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    hor_lines_img = cv2.dilate(img_temp2, hor_kernel, iterations=3)

    cv2.imshow("img", hor_lines_img)
    cv2.waitKey(0)

    alpha = 0.5

    img_final_bin = cv2.addWeighted(ver_lines_img, alpha, hor_lines_img, 1.0 - alpha, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=5)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imshow("img", img_final_bin)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (contours, bboxes) = sort_contours(contours, method="top-to-bottom")

    img_bin = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)

    i = 0

    if not os.path.exists("data/test/boxes/%s/" % name.strip(".png")):
        os.mkdir("data/test/boxes/%s/" % name.strip(".png"))

    print(img.shape)

    res_contours = []

    img_bin = 255 - img_bin

    for c in contours:
        i += 1
        x, y, w, h = cv2.boundingRect(c)
        if w < img.shape[0] / 3:
            cv2.rectangle(img_bin, (x, y), (x + w, y + h), (0, 0, 255))
            cv2.imwrite("data/test/%s/%s.png" % (name.strip(".png"), i), img[y: y + h, x: x + w])
            res_contours.append((x, y, w, h))

    cv2.imshow("img", img_bin)
    cv2.waitKey(0)

    return res_contours


if __name__ == '__main__':
    cut_boxes("15.png")