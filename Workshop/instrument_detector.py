from pathlib import Path
import cv2 as cv
import numpy as np
import json


# Feature space
class Features:
    # function for saving new contours into the training data
    def __init__(self, contour, img):

        # Calculate feature properties
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)

        (min_x, min_y), (min_width, min_height), min_angle = cv.minAreaRect(contour)
        (circle_x, circle_y), circle_radius = cv.minEnclosingCircle(contour)

        self.circularity = (4 * np.pi * area / (perimeter**2))
        self.compactness = (area / (min_width * min_height))
        self.elongation = (min(min_width, min_height) / max(min_width, min_height))
        self.thiness = ((4 * np.pi * area) / (perimeter**2))

        # Intensity found in the image
        mask = np.zeros(img.shape, np.uint8)
        cv.drawContours(mask, [cnt], 0, 255, -1)
        self.intensity = ((cv.mean(img, mask=mask)[0])/255)     # Divide by 255 to normalise

    def return_vector(self):
        feature_vector = []
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
    feature_data = ['circularity', 'compactness', 'elongation', 'thiness', 'intensity']
    instrument_data = []
    for instrument in instruments:
        instrument_data.append(json.load(open(f'data_{instrument}.json', 'r')))
    guitar_distance = 0
    bass_distance = 0
    trumpet_distance = 0

    for num in range(len(feature_data)):
        guitar_distance += pow(test_data[num]-np.array(instrument_data[0][feature_data[num]]), 2)
        bass_distance += pow(test_data[num]-np.array(instrument_data[1][feature_data[num]]), 2)
        trumpet_distance += pow(test_data[num] - np.array(instrument_data[2][feature_data[num]]), 2)

    guitar_distance = min(np.sqrt(guitar_distance))
    bass_distance = min(np.sqrt(bass_distance))
    trumpet_distance = min(np.sqrt(trumpet_distance))

    if guitar_distance < bass_distance and guitar_distance < trumpet_distance:
        return "guitar"
    elif bass_distance < guitar_distance and bass_distance < trumpet_distance:
        return "bass"
    else:
        return "trumpet"


# importing the pictures
PicNum = 2
n_pics = 8
pictures = np.empty(n_pics, dtype=object)
grey = pictures

for i in range(len(pictures)):
    pictures[i] = cv.imread(Path.cwd().as_posix()+'/materialer/guitar/guitar'+str(i+1)+'.jpg')
    grey[i] = cv.imread(Path.cwd().as_posix() + '/materialer/guitar/guitar' + str(i + 1) + '.jpg', cv.IMREAD_GRAYSCALE)

blur = cv.medianBlur(grey[PicNum], 21)
ret, threshold = cv.threshold(blur, 210, 255, cv.THRESH_BINARY_INV)
closed = close_image(threshold, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)),
                     cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35)))

# Extract contours
contours, hierarchy = cv.findContours(image=closed, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

# Save the contours into the Features class
for cnt in contours:
    f = Features(cnt, grey[PicNum])
    cnt_features = f.return_vector()
    cnt_type = check_distance(cnt_features)
    print(cnt_type)


#resize_image(image_copy, 'threshold', 0.4)
#cv.waitKey(0)
