import cv2
import numpy as np

# Define some things for readability
white = [255,255,255]
WB=0.2
WG=0.2
WR=0.6
contrastScale=0.5

#load image
#---potteplante images---
input_billede = cv2.imread(r'C:\Users\mikip\Documents\Fag_Perception\Lektion_2\materialer\flowers.jpg')

greyscaled_billede = np.zeros(input_billede.shape[0:2], dtype=np.uint8)
threshold_billede = np.zeros(input_billede.shape[0:2], dtype=np.uint8)

#---thermal images---
andor_billede1 = cv2.imread(r'C:\Users\mikip\Documents\Fag_Perception\Lektion_2\materialer\thermal1.png',cv2.IMREAD_GRAYSCALE)
andor_billede2 = cv2.imread(r'C:\Users\mikip\Documents\Fag_Perception\Lektion_2\materialer\thermal2.png',cv2.IMREAD_GRAYSCALE)

andor_output = np.zeros(andor_billede1.shape[0:2], dtype=np.uint8)

#---Einsteiner billede----
einstein_billede = cv2.imread(r'C:\Users\mikip\Documents\Fag_Perception\Lektion_2\materialer\Einstein.tif')

einstein_output = np.zeros(einstein_billede.shape, dtype=np.uint8)

#process image
#----greyscaling image-----
for y, row in enumerate(greyscaled_billede):
    for x, pixel in enumerate(row):
        greyscaled_billede[y, x] = (input_billede[y,x,0]*WB) + (input_billede[y,x,1]*WG) + (input_billede[y,x,2]*WR)

#----thresholding blue image-----
for y, row in enumerate(input_billede):
    for x, pixel in enumerate(row):
        threshold_billede[y, x]=0
        if input_billede[y,x,0] < 100:
            threshold_billede[y,x] = 250

#----and or operations----
for y, row in enumerate(andor_billede1):
    for x, pixel in enumerate(row):
        if andor_billede1[y,x]/2+andor_billede2[y,x]/2 > 110:
            andor_output[y, x]= 250

bitwiseAnd = cv2.bitwise_and(andor_billede1,andor_billede2)
bitwiseOr = cv2.bitwise_or(andor_billede1,andor_billede2)

#---Einstein---
for y, row in enumerate(einstein_billede):
    for x, pixel in enumerate(row):
        i = 0
        while i < 3:
            einstein_output[y,x,i] = einstein_billede[y,x,i]*contrastScale
            i += 1

#mask=cv2.inRange(input_billede, np.array([0,10,0]), np.array([100,255,100]))        
#input_billede[mask>0] = white

#show image
cv2.imshow("sejt" , input_billede)
cv2.imshow("sejere" , greyscaled_billede)
cv2.imshow("sejest" , threshold_billede)
cv2.imshow("andor" , andor_output)
cv2.imshow("andor" , bitwiseAnd)
cv2.imshow("or" , bitwiseOr)
cv2.imshow("Einstein before" , einstein_billede)
cv2.imshow("Einstein after" , einstein_output)
cv2.waitKey(0)