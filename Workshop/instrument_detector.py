from pathlib import Path
import cv2 as cv
import numpy as np
from collections import deque
from dataclasses import dataclass


# Feature space
@dataclass
class Features:
    contourIndex = int
    area = int
    perimeter = int
    circularity = float
    hasHoles = bool
    elongation = float

# A function that allows me to display an image an resize it
def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0],image.shape[1]]
    [height, width] = [procent*height, procent*width]
    cv.namedWindow(image_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(image_name, int(width), int(height))
    cv.imshow(image_name, image)


# importing the pictures
PicNum = 1
n_pics = 8
pictures = np.empty(n_pics, dtype=object)
grey = pictures

for i in range(len(pictures)):
    pictures[i] = cv.imread(Path.cwd().as_posix()+'/materialer/trumpet/trumpet'+str(i+1)+'.jpg')
    grey[i] = cv.imread(Path.cwd().as_posix() + '/materialer/trumpet/trumpet' + str(i + 1) + '.jpg', cv.IMREAD_GRAYSCALE)

ret, threshold = cv.threshold(grey[1], 210, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(image=threshold, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
image_copy = pictures[1].copy()
cv.drawContours(image_copy, contours, -1, (0, 255, 0), 3)
for i in range(len(contours)):
    cnt = contours[i]



#     f.contourIndex = i
#     f.area = cv.contourArea(cnt)
#     f.perimeter = cv.arcLength(cnt, True)
#     f.circularity = (4 * np.pi * f.area) / pow(f.perimeter, 2)
#
#     RotatedRect, box = cv.minAreaRect(cnt)
#     f.elongation = max(box.size.width / box.size.height, box.size.height / box.size.width)

resize_image(image_copy, 'threshold', 0.4)
cv.waitKey(0)
