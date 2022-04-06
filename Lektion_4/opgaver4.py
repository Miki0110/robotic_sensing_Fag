import numpy as np
import cv2

opgave = 4

#Tag dette binære billede:

if opgave == 1:
    input = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
             [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
             [0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]]
    #og et 3x3 firkantet strukturelement:
    strucElem = [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]

    # erosion
    output =[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    print("erosion = \n", np.array(output))
    # dilation
    output = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]]
    print("dilation = \n", np.array(output))
    # closing
    output = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
             [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]]
    print("closing = \n", np.array(output))
    # opening
    output =[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    print("opening = \n", np.array(output))

#Brug morphology og image subtraction til at finde kanten/omkreds af hver prik i billedet "dots-bin.png".
if opgave == 2:
    dots = cv2.imread(r'C:\Users\Muku\Documents\robotic_sensing_Fag\Lektion_4\materialer\dots-bin.png', cv2.IMREAD_GRAYSCALE)
    ret, dotsBin = cv2.threshold(dots, 30, 255, cv2.THRESH_BINARY)

    strucElem = np.ones((4, 4), dtype=int)

    erodeDots = cv2.erode(dotsBin, strucElem)
    finalDots = cv2.subtract(dotsBin, erodeDots)

    cv2.imshow("original", dotsBin)
    cv2.imshow("eroded", erodeDots)
    cv2.imshow("Final", finalDots)
    cv2.waitKey(0)

#Brug morphology til at fjerne linjerne og beholde prikkerne i billedet "dots-lines.png".
if opgave == 3:
    dots = cv2.imread(r'C:\Users\Muku\Documents\robotic_sensing_Fag\Lektion_4\materialer\dots-lines.png',
                      cv2.IMREAD_GRAYSCALE)
    ret, dotsBin = cv2.threshold(dots, 30, 255, cv2.THRESH_BINARY)

    #   by making a thick structure element the lines will not pass the erosion
    strucElem = np.ones((8, 10), dtype=int)

    erodeDots = cv2.erode(dotsBin, strucElem)
    finalDots = cv2.dilate(erodeDots, strucElem)

    cv2.imshow("original", dotsBin)
    cv2.imshow("eroded", erodeDots)
    cv2.imshow("Final", finalDots)
    cv2.waitKey(0)
#Lav et lille program der kan segmentere æblet i billedet "apple.jpg". Det røde "kød" på æblet skal blive hvidt, og resten, inkl. stilk og blad, skal blive sort.
if opgave == 4:
    apple = cv2.imread(r'C:\Users\mikip\Documents\Fag_Perception\Lektion_4\materialer\apple.jpg')
    hsvApple = cv2.cvtColor(apple, cv2.COLOR_BGR2HSV)

    # first cut out the apple part
    upperLimit1 = np.array([180, 255, 255])
    lowerLimit1 = np.array([160, 80, 0])

    upperLimit2 = np.array([15, 255, 255])
    lowerLimit2 = np.array([0, 80, 0])

    mask1 = cv2.inRange(hsvApple, lowerLimit1,  upperLimit1)
    mask2 = cv2.inRange(hsvApple, lowerLimit2,  upperLimit2)
    appleBin = mask1+mask2

    # then use closing method to remove the hole and stalk
    strucElem1 = np.ones((5, 25), dtype=int)
    # since the hole has a different shape than the stalk I am using a different SE for dilation
    strucElem2 = np.ones((25, 25), dtype=int)

    erodeApple = cv2.erode(appleBin, strucElem1)
    finalApple = cv2.dilate(erodeApple, strucElem2)

    cv2.imshow("original", apple)
    cv2.imshow("Binary", appleBin)
    cv2.imshow("erode", erodeApple)
    cv2.imshow("Final", finalApple)
    cv2.waitKey(0)