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
        feature_vector.append(self.intensity)
        return feature_vector


# A function that allows me to display an image and resize it
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
def check_distance(test_data, k_size):
    # TODO FIX FALSE POSITIVES
    # defining instruments, and the features
    instruments = ['guitar', 'bass', 'trumpet', 'drumm']
    feature_data = ['holes', 'circularity', 'compactness', 'elongation', 'intensity']
    instrument_data = []
    dist = []

    # Load data from training set
    for instrument in instruments:
        instrument_data.append(json.load(open(f'./training_data/data_{instrument}.json', 'r')))

    # Calculate nearest neighbour
    for i in range(len(instruments)):
        temp_distance = 0
        for num in range(len(feature_data)):
            temp_distance += (pow(test_data[num]-np.array(instrument_data[i][feature_data[num]], dtype='float64'), 2))
        # square that
        dist.append(np.sqrt(temp_distance))
    # need my distances in a vector not an array
    dist = np.concatenate(dist, axis=None)

    # Checking to see if the contour is an instrument, return if it is not
    sorted_dist = np.sort(dist, axis=None)
    if sorted_dist[0]+sorted_dist[1]+sorted_dist[2] >= 2:
        return 0

    # check the k-size smallest distances and save their index
    maxvalues_index = []
    for i in range(k_size):
        maxvalues_index.append(np.where(dist == sorted_dist[i])[0][0])
    # compare the index to the instrument number and count the amounts in each
    a = []
    for i in range(len(instruments)):
        a.append(sum(1 for n in maxvalues_index if n > (i) * 15 - 1 and n <= (i + 1) * 15 - 1))
    return instruments[a.index(max(a))]


# define amount of pictures to import
n_pics = 19

for i in range(n_pics):

    # Read folder and import images
    img = cv.imread(f'./materialer/{str(i + 1)}.jpg')
    grey = cv.imread(f'./materialer/{str(i + 1)}.jpg', cv.IMREAD_GRAYSCALE)
    image_copy = img.copy()

    # Image processing
    # TODO apply to a real scenario instead of a white background
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
    for n in range(len(contours)):
        cnt = contours[n]
        # Make sure the contour has an area and is "parentless" :(
        if cv.contourArea(cnt) > threshold.shape[0]*threshold.shape[1]*0.01 and np.array(hierarchy[n][3] == -1):
            # If statement looking at how many children the contour has
            holes = hierarchy[np.array(hierarchy[:][:, 3] == n)]

            # Save data into the feature class
            f = Features(cnt, len(holes), grey)
            cnt_features = f.return_vector()
            # compare nearest neighbour to training data
            cnt_type = check_distance(cnt_features, 3)
            if cnt_type != 0:
                # Draw and show results
                x, y, w, h = cv.boundingRect(cnt)
                cv.drawContours(image_copy, cnt, -1, (255, 100, 100), 3)
                cv.putText(image_copy, f'{cnt_type}', (x+int(w/2)-int(12*len(cnt_type)), y+int(h/2)), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3, cv.LINE_AA)
            else:
                cv.drawContours(image_copy, cnt, -1, (20, 20, 150), 2)


    resize_image(image_copy, f'Instrument{i+1} result', 0.8)
    cv.waitKey(0)
    cv.destroyWindow(f'Instrument{i + 1} result')


