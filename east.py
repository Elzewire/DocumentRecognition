from imutils.object_detection import non_max_suppression
from preprocess import prepare
import numpy as np
import cv2
import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to input image")
ap.add_argument("-e", "--east", type=str, help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320, help="resized image width (should be multiple of 32)")
ap.add_argument("-g", "--height", type=int, default=320, help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

image = prepare(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

(nh, nw) = (args["height"], args["width"])
rh = h / float(nh)
rw = w / float(nw)

image = cv2.resize(image, (nw, nh))
(h, w) = image.shape[:2]

cv2.imshow("txt", image)
cv2.waitKey(0)



layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
net = cv2.dnn.readNet(args["east"])

blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layer_names)
end = time.time()

print("Text detection finished in %ss" % (end - start))

(n_rows, n_cols) = scores.shape[2:4]
rects = []
conf = []

for y in range(n_rows):
    scores_data = scores[0, 0, y]
    x_data0 = geometry[0, 0, y]
    x_data1 = geometry[0, 1, y]
    x_data2 = geometry[0, 2, y]
    x_data3 = geometry[0, 3, y]
    angles_data = geometry[0, 4, y]
    for x in range(n_cols):
        if scores_data[x] >= args["min_confidence"]:
            (offset_x, offset_y) = (x * 4.0, y * 4.0)

            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            end_x = int(offset_x + (cos * x_data1[x])) + (sin * x_data2[x])
            end_y = int(offset_y + (sin * x_data1[x])) + (cos * x_data2[x])
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            rects.append((start_x, start_y, end_x, end_y))
            conf.append(scores_data[x])

boxes = non_max_suppression(np.array(rects), probs=conf)

for (start_x, start_y, end_x, end_y) in boxes:
    start_x = int(start_x * rw)
    start_y = int(start_y * rh)
    end_x = int(end_x * rw)
    end_y = int(end_y * rh)

    cv2.rectangle(orig, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

cv2.imshow("txt", orig)
cv2.waitKey(0)