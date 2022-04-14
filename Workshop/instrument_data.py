from pathlib import Path
import cv2 as cv
import numpy as np
import json
from scipy.io import savemat

# Simple function for diplaying image
def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0], image.shape[1]]
    [height, width] = [procent * height, procent * width]
    cv.namedWindow(image_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(image_name, int(width), int(height))
    cv.imshow(image_name, image)
    return

# Simple functions for morphology
def open_image(input_image, e_kernel, d_kernel):
    e_out = cv.erode(input_image, e_kernel)
    d_out = cv.dilate(e_out, d_kernel)
    return d_out
def close_image(input_image, e_kernel, d_kernel):
    d_out = cv.dilate(input_image, d_kernel)
    e_out = cv.erode(d_out, e_kernel)
    return e_out


class Features:
    # startup for initialising the features
    def __init__(self, instrument_type):
        self.type = instrument_type
        self.circularity = []
        self.compactness = []
        self.elongation = []
        self.thiness = []
        self.intensity = []

    # function for saving new contours into the training data
    def save_cnt(self, contour, img):

        # Calculate feature properties
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)

        (min_x, min_y), (min_width, min_height), min_angle = cv.minAreaRect(contour)
        (circle_x, circle_y), circle_radius = cv.minEnclosingCircle(contour)

        self.circularity.append(4 * np.pi * area / (perimeter**2))
        self.compactness.append(area / (min_width * min_height))
        self.elongation.append(min(min_width, min_height) / max(min_width, min_height))
        self.thiness.append((4 * np.pi * area) / (perimeter**2))

        # Intensity found in the image
        mask = np.zeros(img.shape, np.uint8)
        cv.drawContours(mask, [cnt], 0, 255, -1)
        self.intensity.append((cv.mean(img, mask=mask)[0])/255)     # Divide by 255 to normalise

    # used for analysing the data extracted from training data
    def export_matlab(self, file_name):
        savemat(file_name, {f'circularity': self.circularity, f'compactness': self.compactness,
                            f'elongation': self.elongation, f'thiness': self.thiness, 'intensity': self.intensity})


# importing the pictures
instruments = ['guitar', 'bass', 'trumpet']
n_pics = 8

for instrument in instruments:
    # Initiate json files and Features class
    outfile = open(f'data_{instrument}.json', 'w')
    f = Features(instrument)
    for i in range(n_pics):
        picture = cv.imread(f'{Path.cwd().as_posix()}/materialer/{instrument}/{instrument}{i+1}.jpg')
        grey = cv.imread(f'{Path.cwd().as_posix()}/materialer/{instrument}/{instrument}{i+1}.jpg', cv.IMREAD_GRAYSCALE)

        # Image processing
        blur = cv.medianBlur(grey, 21)
        ret, threshold = cv.threshold(blur, 210, 255, cv.THRESH_BINARY_INV)
        closed = close_image(threshold, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)),
                             cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35)))

        # Extract contours
        contours, hierarchy = cv.findContours(image=closed, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

        # Save the contours into the Features class
        for cnt in contours:
            f.save_cnt(cnt, grey)

    # write all the data into the .json and .mat files
    jsonStr = json.dumps(f.__dict__, indent=4)
    outfile.write(jsonStr)
    f.export_matlab(f'data_{instrument}.mat')

