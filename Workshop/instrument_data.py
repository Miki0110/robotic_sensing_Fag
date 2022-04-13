from pathlib import Path
import cv2 as cv
import numpy as np
import json

# Simple function for diplaying image
def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0], image.shape[1]]
    [height, width] = [procent * height, procent * width]
    cv.namedWindow(image_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(image_name, int(width), int(height))
    cv.imshow(image_name, image)
    return


def open_image(input_image, e_kernel, d_kernel):
    e_out = cv.erode(input_image, e_kernel)
    d_out = cv.dilate(e_out, d_kernel)
    return d_out


def close_image(input_image, e_kernel, d_kernel):
    d_out = cv.dilate(input_image, d_kernel)
    e_out = cv.erode(d_out, e_kernel)
    return e_out


class Features:
    def __init__(self, contour, img):

        # Calculate feature properties
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)

        (min_x, min_y), (min_width, min_height), min_angle = cv.minAreaRect(contour)
        #(ellipse_x, ellipse_y), (ellipse_MA, ellipse_ma), ellipse_angle = cv.fitEllipse(contour)
        (circle_x, circle_y), circle_radius = cv.minEnclosingCircle(contour)

        self.circularity = 4 * np.pi * area / (perimeter**2)
        self.compactness = area / (min_width * min_height)
        self.elongation = min(min_width, min_height) / max(min_width, min_height)
        #self.fatness = ellipse_ma / ellipse_MA
        self.thiness = (4 * np.pi * area) / (perimeter**2)

        # Intensity found in the image
        mask = np.zeros(img.shape, np.uint8)
        cv.drawContours(mask, [cnt], 0, 255, -1)
        self.intensity = cv.mean(img, mask=mask)[0]


# importing the pictures
instruments = ['guitar', 'bass', 'trumpet']
n_pics = 8

for instrument in instruments:
    outfile = open(f'data_{instrument}.json', 'w')
    for i in range(n_pics):
        picture = cv.imread(f'{Path.cwd().as_posix()}/materialer/{instrument}/{instrument}{i+1}.jpg')
        grey = cv.imread(f'{Path.cwd().as_posix()}/materialer/{instrument}/{instrument}{i+1}.jpg', cv.IMREAD_GRAYSCALE)

        blur = cv.medianBlur(grey, 21)
        ret, threshold = cv.threshold(blur, 210, 255, cv.THRESH_BINARY_INV)
        closed = close_image(threshold, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)),cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35)))
        contours, hierarchy = cv.findContours(image=closed, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

        # image_copy = picture.copy()
        # cv.drawContours(image_copy, contours, -1, (0, 255, 0), 3)

        for n in range(len(contours)):
            cnt = contours[n]
            f = Features(cnt, grey)
            jsonStr = json.dumps(f.__dict__, indent=4)
        outfile.write(jsonStr)

