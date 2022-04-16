from pathlib import Path
import cv2 as cv
import numpy as np
import json


# Feature space
class Features:
    # function for saving new contours into the training data
    def __init__(self, contour, n_children, img):
        # Save the amount of holes
        self.holes = n_children/10      # Somewhat normalised

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

    # function creating a vector, used for calculating distance
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
    # defining instruments, and the features
    instruments = ['guitar', 'bass', 'trumpet', 'drumm']
    feature_data = ['holes', 'circularity', 'compactness', 'elongation', 'thiness', 'intensity']
    instrument_data = []
    guitar_distance,bass_distance,trumpet_distance,drumm_distance = 0,0,0,0

    # Load data from training set
    for instrument in instruments:
        instrument_data.append(json.load(open(f'data_{instrument}.json', 'r')))

    # Calculate nearest neighbour
    for num in range(len(feature_data)):
        guitar_distance += (pow(test_data[num]-np.array(instrument_data[0][feature_data[num]], dtype='float64'), 2))
        bass_distance += (pow(test_data[num]-np.array(instrument_data[1][feature_data[num]], dtype='float64'), 2))
        trumpet_distance += (pow(test_data[num] - np.array(instrument_data[2][feature_data[num]], dtype='float64'), 2))
        drumm_distance += (pow(test_data[num] - np.array(instrument_data[3][feature_data[num]], dtype='float64'), 2))

    guitar_distance = min(np.sqrt(guitar_distance))
    bass_distance = min(np.sqrt(bass_distance))
    trumpet_distance = min(np.sqrt(trumpet_distance))
    drumm_distance = min(np.sqrt(drumm_distance))

    # Return the closest value
    if guitar_distance < bass_distance and guitar_distance < trumpet_distance and guitar_distance < drumm_distance:
        print(f'guitar_distance: {guitar_distance}')
        return "guitar"
    elif bass_distance < guitar_distance and bass_distance < trumpet_distance and bass_distance < drumm_distance:
        print(f'bass_distance: {bass_distance}')
        return "bass"
    elif trumpet_distance < guitar_distance and trumpet_distance < bass_distance and trumpet_distance < drumm_distance:
        print(f'trumpet_distance: {trumpet_distance}')
        return "trumpet"
    else:
        print(f'drumm_distance: {drumm_distance}')
        return 'drumm'


# define amount of pictures to import
n_pics = 8

for i in range(n_pics):
    # Read folder and import images
    img = cv.imread(f'{Path.cwd().as_posix()}/materialer/{str(i + 1)}.jpg')
    grey = cv.imread(f'{Path.cwd().as_posix()}/materialer/{str(i + 1)}.jpg', cv.IMREAD_GRAYSCALE)
    image_copy = img.copy()

    # Image processing
    hsvImg = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    blur = cv.medianBlur(hsvImg, 7)
    # HSL thresholding is used due to the background always being white
    # a different method is needed if live footage was used
    threshold = cv.inRange(blur, np.array([0, 0, 0]), np.array([255, 225, 255]))

    # Using morphology to remove noise (mostly due to watermarks)
    opened = open_image(threshold, cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10)),
                        cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10)))

    # Extract contours
    contours, hierarchy = cv.findContours(image=threshold, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]

    # Find the largest contour
    outer = 0
    largest_a = 0
    for n in range(len(hierarchy)):
        if np.array(hierarchy[n][3] == -1):
            a = cv.contourArea(contours[n])
            if largest_a < a:
                largest_a = a
                outer = n
    cnt = contours[outer]

    # If statement looking at how many children the contour has
    holes = hierarchy[np.array(hierarchy[:][:, 3] == outer)]
    print(f'holes: {len(holes)}')

    # Save data into the feature class
    f = Features(cnt, len(holes), grey)
    cnt_features = f.return_vector()
    # compare nearest neighbour to training data
    cnt_type = check_distance(cnt_features)

    # Draw and show results
    cv.drawContours(image_copy, cnt, -1, (255, 0, 0), 3)
    cv.putText(image_copy, f'{cnt_type}', (15,35), cv.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3, cv.LINE_AA)
    resize_image(image_copy, f'Instrument{i+1} result', 0.8)

cv.waitKey(0)
