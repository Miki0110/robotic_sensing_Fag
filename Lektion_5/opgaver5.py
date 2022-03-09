import cv2
from collections import deque

import numpy as np


def grassfire_transform(img):
    burn_num = 1
    grassfire = np.zeros_like(img, dtype=np.uint8)
    for y, row in enumerate(img):
        for x, pixel in enumerate(row):
            if grassfire[y, x] == 0:
                if connectivity(img, [y, x], burn_num, grassfire) == 1:
                    burn_num += 1
            else:
                print("nice")
    print(burn_num)
    return grassfire


def connectivity(img, coordinate, burn_num, grassfire):
    burn_queue = deque()
    succ = 0
    if img[coordinate[0], coordinate[1]] == 255:
        burn_queue.append(coordinate)
        succ = 1
    while len(burn_queue) > 0:
        coordinate = burn_queue.pop()
        grassfire[coordinate[0], coordinate[1]] = burn_num

        if img[coordinate[0], coordinate[1] - 1] == 255:
            burn_queue.append([coordinate[0], coordinate[1] - 1])
            print("yup")

        if img[coordinate[0] - 1, coordinate[1]] == 255:
            burn_queue.append([coordinate[0] - 1, coordinate[1]])
            print("yup2")

        if img[coordinate[0], coordinate[1]] + 1 == 255:
            burn_queue.append([coordinate[0], coordinate[1] + 1])
            print("yup3")

        if img[coordinate[0], coordinate[1]] + 1 == 255:
            burn_queue.append([coordinate[0], coordinate[1] + 1])
            print("yup4")

    return succ


# Find de fire objekter i billedet 'tools-black.jpg' (output: fire hvide BLOBs, sort baggrund)
full_image = cv2.imread(r'C:\Users\mikip\Documents\Fag_Perception\Lektion_5\materialer\tools-black.jpg',
                        cv2.IMREAD_GRAYSCALE)

medianBlur = cv2.medianBlur(full_image, 15)
ret, threshold = cv2.threshold(full_image, 115, 255, cv2.THRESH_BINARY)
full_image = cv2.imread(r'C:\Users\mikip\Documents\Fag_Perception\Lektion_5\materialer\Untitled.png',
                        cv2.IMREAD_GRAYSCALE)

newimg = grassfire_transform(full_image)
# # størrelser
# scale = 0.3
# delta = 1
# ddepth = cv2.CV_16S
# # Gradient-X
# grad_x = cv2.Sobel(threshold, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
# # Gradient-Y
# grad_y = cv2.Sobel(threshold, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
#
# abs_grad_x = cv2.convertScaleAbs(grad_x)
# abs_grad_y = cv2.convertScaleAbs(grad_y)
#
# grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

#cv2.imshow("original", full_image)
#cv2.imshow("median", medianBlur)
#cv2.imshow("Threshold", threshold)
cv2.imshow("new", newimg)
# cv2.imshow("sobel", grad)
# cv2.imshow("Final", finalApple)
cv2.waitKey(0)

# I et nyt billede, tegn konturerne af alle objekterne i hver sin farve


# Vælg nogle relevante features der kan bruges til at klassificere hvert objekt. Udtræk disse features


# Lav et simpelt klassifikationssystem, så du ved at sætte threshold-værdier på hver feature kan vise ét objekt ad gangen


# Simpelt eksempel: Vis den BLOB der har større areal end 100 pixels og lavere circularity end 0,5


# Eksperimenter med forskellige features
