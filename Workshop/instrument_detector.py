from pathlib import Path
import cv2 as cv
import numpy as np
import json
from collections import deque


# Feature space
class Features:
    # function for saving new contours into the training data
    def __init__(self, contour, n_children, img):
        self.holes = n_children

        # Calculate feature properties
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)

        (min_x, min_y), (min_width, min_height), min_angle = cv.minAreaRect(contour)
        (circle_x, circle_y), circle_radius = cv.minEnclosingCircle(contour)

        self.circularity = (4 * np.pi * area / (perimeter**2))
        self.compactness = (area / (min_width * min_height))
        self.elongation = (min(min_width, min_height) / max(min_width, min_height)*2)
        self.thiness = ((4 * np.pi * area) / (perimeter**2)*2)

        # Intensity found in the image
        mask = np.zeros(img.shape, np.uint8)
        cv.drawContours(mask, [cnt], 0, 255, -1)
        self.intensity = ((cv.mean(img, mask=mask)[0])/255)     # Divide by 255 to normalise

    def return_vector(self):
        feature_vector = []
        feature_vector.append(self.holes)
        feature_vector.append(self.circularity)
        feature_vector.append(self.compactness)
        feature_vector.append(self.elongation)
        feature_vector.append(self.thiness)
        feature_vector.append(self.intensity)
        return feature_vector


# A function that allows me to display an image an resize it
def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0],image.shape[1]]
    [height, width] = [procent*height, procent*width]
    cv.namedWindow(image_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(image_name, int(width), int(height))
    cv.imshow(image_name, image)


# Simple functions for morphology
def open_image(input_image, e_kernel, d_kernel):
    e_out = cv.erode(input_image, e_kernel)
    d_out = cv.dilate(e_out, d_kernel)
    return d_out
def close_image(input_image, e_kernel, d_kernel):
    d_out = cv.dilate(input_image, d_kernel)
    e_out = cv.erode(d_out, e_kernel)
    return e_out


# Check distance
def check_distance(test_data):
    instruments = ['guitar', 'bass', 'trumpet']
    feature_data = ['holes', 'circularity', 'compactness', 'elongation', 'thiness', 'intensity']
    instrument_data = []
    guitar_distance = 0
    bass_distance = 0
    trumpet_distance = 0

    for instrument in instruments:
        instrument_data.append(json.load(open(f'data_{instrument}.json', 'r')))

    for num in range(len(feature_data)):
        guitar_distance += (pow(test_data[num]-np.array(instrument_data[0][feature_data[num]], dtype='float64'), 2))
        bass_distance += (pow(test_data[num]-np.array(instrument_data[1][feature_data[num]], dtype='float64'), 2))
        trumpet_distance += (pow(test_data[num] - np.array(instrument_data[2][feature_data[num]], dtype='float64'), 2))
    print(sorted(np.sqrt(guitar_distance)))
    print(sorted(np.sqrt(bass_distance)))
    print(sorted(np.sqrt(trumpet_distance)))
    guitar_distance = min(np.sqrt(guitar_distance))
    bass_distance = min(np.sqrt(bass_distance))
    trumpet_distance = min(np.sqrt(trumpet_distance))

    for i in range(3):
        if guitar_distance < bass_distance and guitar_distance < trumpet_distance:
            print(f'guitar_distance: {guitar_distance}')
            return "guitar"
        elif bass_distance < guitar_distance and bass_distance < trumpet_distance:
            print(f'bass_distance: {bass_distance}')
            return "bass"
        else:
            print(f'trumpet_distance: {trumpet_distance}')
            return "trumpet"


# importing the pictures
n_pics = 6
pictures = []
grey = []
image_copy = []

for i in range(n_pics):
    pictures.append(cv.imread(f'{Path.cwd().as_posix()}/materialer/{str(i + 1)}.jpg'))
    grey.append(cv.imread(f'{Path.cwd().as_posix()}/materialer/{str(i + 1)}.jpg', cv.IMREAD_GRAYSCALE))

    # Image processing
    hsvImg = cv.cvtColor(pictures[i], cv.COLOR_BGR2HLS)
    blur = cv.medianBlur(hsvImg, 21)
    threshold = cv.inRange(blur, np.array([0, 0, 0]), np.array([255, 235, 255]))

    opened = open_image(threshold, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)),
                        cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)))

    # Extract contours
    contours, hierarchy = cv.findContours(image=opened, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    image_copy.append(pictures[i].copy())
    hierarchy = hierarchy[0]
    outer = hierarchy[np.array(hierarchy[:][:, 3] == -1)]

    # Save the contours into the Features class
    for heir in outer:
        # find the parentless contours
        cnt = contours[abs(heir[0]) - 1]
        # write down the children attached to it
        inner = hierarchy[np.array(hierarchy[:][:, 3] == abs(heir[0]) - 1)]
        print(f'number: {i+1}, children: {len(inner)}')
        f = Features(cnt, len(inner), grey[i])
        cnt_features = f.return_vector()
        cnt_type = check_distance(cnt_features)

        cv.drawContours(image_copy[i], cnt, 3, (255, 0, 0), 1)
        cv.putText(image_copy[i], f'{cnt_type}', (15,35), cv.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3, cv.LINE_AA)
        resize_image(image_copy[i], f'Instrument{i+1} result', 0.8)

cv.waitKey(0)
