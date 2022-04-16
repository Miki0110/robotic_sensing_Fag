from pathlib import Path
import cv2 as cv
import numpy as np
import json
from scipy.io import savemat

###############
#  DEBUGGING  #
DEBUG = 0
###############

# Simple function for displaying image
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
        self.holes = []
        self.circularity = []
        self.compactness = []
        self.elongation = []
        self.thiness = []
        self.intensity = []

    # function for saving new contours into the training data
    def save_cnt(self, contour, n_children, img):

        # Calculate feature properties
        self.holes.append(n_children/10)
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)

        (min_x, min_y), (min_width, min_height), min_angle = cv.minAreaRect(contour)
        (circle_x, circle_y), circle_radius = cv.minEnclosingCircle(contour)

        self.circularity.append(4 * np.pi * area / (perimeter**2))
        self.compactness.append(area / (min_width * min_height))
        self.elongation.append(min(min_width, min_height) / max(min_width, min_height)*2)
        self.thiness.append((4 * np.pi * area) / (perimeter**2)*2)

        # Intensity found in the image
        mask = np.zeros(img.shape, np.uint8)
        cv.drawContours(mask, [cnt], 0, 255, -1)
        self.intensity.append((cv.mean(img, mask=mask)[0])/255)     # Divide by 255 to normalise

    # used for analysing the data extracted from training data
    def export_matlab(self, file_name):
        savemat(file_name, {f'holes': self.holes, f'circularity': self.circularity, f'compactness': self.compactness,
                            f'elongation': self.elongation, f'thiness': self.thiness, 'intensity': self.intensity})


# Define instruments and how much training data is available
instruments = ['guitar', 'bass', 'trumpet', 'drumm']
n_pics = 15

for instrument in instruments:
    # Initiate json files and Features class
    outfile = open(f'data_{instrument}.json', 'w')
    f = Features(instrument)

    # Go through the training data 1 by 1
    for i in range(n_pics):
        picture = cv.imread(f'{Path.cwd().as_posix()}/materialer/training_data/{instrument}/{instrument}{i+1}.jpg')
        grey = cv.imread(f'{Path.cwd().as_posix()}/materialer/training_data/{instrument}/{instrument}{i+1}.jpg', cv.IMREAD_GRAYSCALE)

        # Image processing
        hsvImg = cv.cvtColor(picture, cv.COLOR_BGR2HLS)
        blur = cv.medianBlur(hsvImg, 7)
        threshold = cv.inRange(blur, np.array([0, 0, 0]), np.array([255, 225, 255]))

        # Using morphology to remove the worst noise
        opened = open_image(threshold, cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10)),
                             cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10)))

        # Extract contours
        contours, hierarchy = cv.findContours(image=threshold, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
        hierarchy = hierarchy[0]

        # Finding the largest contour
        outer = 0
        largest_a = 0
        for n in range(len(hierarchy)):
            if np.array(hierarchy[n][3] == -1):
                a = cv.contourArea(contours[n])
                if largest_a < a:
                    largest_a = a
                    outer = n
        cnt = contours[outer]

        # Simple if statement to find children related to the contour
        holes = hierarchy[np.array(hierarchy[:][:, 3] == outer)]

        # Save the data into the Feature class
        f.save_cnt(cnt, len(holes), grey)
        # Draw result for debugging
        if DEBUG:
            cv.drawContours(picture, cnt, -1, (255, 0, 0), 5)
            resize_image(picture, f'Instrument{i + 1} result', 0.8)
            cv.waitKey(0)
            cv.destroyWindow(f'Instrument{i + 1} result')

    # write all the data into the .json and .mat files
    jsonStr = json.dumps(f.__dict__, indent=4)
    outfile.write(jsonStr)
    f.export_matlab(f'data_{instrument}.mat')

