import cv2 as cv
from pathlib import Path
import numpy as np
from scipy.interpolate import splprep, splev

# Select picture path from .../P4-Automatic_Inspection_of_sewers/ directory
picPath = '/materials/GR-type_0_2.jpg'

# Advanced version of Thinness ratio
def find_thinness(cnt):
    # Requires the scipy to smooth out the contour
    contour = cnt[:, 0, :]

    # smooth contour
    tck, u = splprep(contour.T, u=None, s=30, per=1)
    u_new = np.linspace(u.min(), u.max(), 50)
    new_cont = np.column_stack(splev(u_new, tck, der=0))

    # change types to stop cv2 complaining
    new_cont = new_cont.astype(np.float32)

    # calculate as before
    area = cv.contourArea(new_cont)
    circum = cv.arcLength(new_cont, True)
    thinness = (4 * np.pi * area) / (circum**2)
    return thinness

# Simple version of thinness
def find_thinness_s(cnt):
    # compute the area of the contour along with the bounding box
    # to compute the aspect ratio
    area = cv.contourArea(cnt)
    circum = cv.arcLength(cnt, True)

    thinness = (4 * np.pi * area) / (circum**2)
    return thinness

def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0],image.shape[1]]
    [height, width] = [procent*height, procent*width]
    cv.namedWindow(image_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(image_name, int(width), int(height))
    cv.imshow(image_name, image)

# read image
grey_img = cv.imread(f'{Path.cwd().parent.as_posix()}/Lektion_5/materialer/tools-black.jpg',
                        cv.IMREAD_GRAYSCALE)
color_img = cv.imread(f'{Path.cwd().parent.as_posix()}/Lektion_5/materialer/tools-black.jpg')


medianBlur = cv.medianBlur(grey_img, 15)
ret, threshold = cv.threshold(medianBlur, 150, 255, cv.THRESH_BINARY)

contours, hierarchy = cv.findContours(image=threshold, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
img_copy1 = color_img.copy()
img_copy2 = color_img.copy()

# data from contours
for cnt in contours:
    # epsilon = 0.01 * cv.arcLength(cnt, True)
    # approx = cv.approxPolyDP(cnt, epsilon, True)
    x, y = cnt[0][0]
    cv.drawContours(img_copy1, [cnt], 0, (0), 3)
    cv.drawContours(img_copy2, [cnt], 0, (0), 3)
    tn = find_thinness_s(cnt)
    cv.putText(img_copy1, f"thinness: {tn}", (x, y), cv.FONT_HERSHEY_COMPLEX, 1, 0, 2)
    tn = find_thinness_s(cnt)
    cv.putText(img_copy2, f"thinness: {tn}", (x, y), cv.FONT_HERSHEY_COMPLEX, 1, 0, 2)


resize_image(grey_img, 'normal', 0.4)
resize_image(medianBlur, 'blurred', 0.4)
resize_image(threshold, 'threshold', 0.4)
resize_image(img_copy1, 'simple', 0.4)
resize_image(img_copy1, 'fixed', 0.4)
cv.waitKey(0)


