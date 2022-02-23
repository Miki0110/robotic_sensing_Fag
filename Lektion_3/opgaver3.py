import cv2
import numpy as np

#definer show
opgave = 6

#Prøv forskellige filtre med varierende kernestørrelse på dit billede, og se hvilken effekt det har:
#blur(), GaussianBlur(), medianBlur()
if opgave == 1:
    papir = cv2.imread(r'C:\Users\mikip\Documents\Fag_Perception\Lektion_3\materialer\papir.jpg')

    blurPapir=cv2.blur(papir,(5,5))
    gaussPapir=cv2.GaussianBlur(papir,(5,5),0.2)
    medianPapir=cv2.medianBlur(papir,15)

    cv2.imshow("normal", papir)
    cv2.imshow("blur", blurPapir)
    cv2.imshow("gauss", gaussPapir)
    cv2.imshow("median", medianPapir)
    cv2.waitKey(0)
#Vælg den bedste metode til at fjerne støj fra 'board.tif'
if opgave == 2:
    board = cv2.imread(r'C:\Users\mikip\Documents\Fag_Perception\Lektion_3\materialer\board.tif')

    blurBoard=cv2.blur(board, (5, 5))
    gaussBoard=cv2.GaussianBlur(board, (5, 5), 7)
    medianBoard=cv2.medianBlur(board, 7)

    cv2.imshow("normal", board)
    cv2.imshow("blur", blurPapir)
    cv2.imshow("gauss", gaussPapir)
    cv2.imshow("median", medianPapir) #median worked the best trust me
    cv2.waitKey(0)

#Segmenter det sorte skrift fra baggrunden på billedet 'papir.jpg'
    #Brug samme fremgangsmåde som videoeksemplet med ImageJ. Teksten bliver hvid og baggrunden sort.
if opgave == 3:
    papir = cv2.imread(r'C:\Users\mikip\Documents\Fag_Perception\Lektion_3\materialer\papir.jpg', cv2.IMREAD_GRAYSCALE)
    blur = cv2.blur(papir, (221, 221))
    fixedImage = cv2.subtract(blur, papir)

    ret, threshold = cv2.threshold(fixedImage, 25, 255, cv2.THRESH_BINARY)
    cv2.imshow("blur", blur)
    cv2.imshow("fixed", fixedImage)
    cv2.imshow("treshold", threshold)
    cv2.waitKey(0)



#Anvend horisontal og vertikal Sobel kerner til at fremhæve kanterne af dit eget billede (brug funktionen Sobel())

if opgave == 4:
    #størrelser
    scale = 0.3
    delta = 1
    ddepth = cv2.CV_16S

    mit = cv2.imread(r'C:\Users\mikip\Documents\Fag_Perception\Lektion_3\materialer\egetBillede.png', cv2.IMREAD_GRAYSCALE)
    blur = cv2.medianBlur(mit, 9)
    ret, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    # Gradient-X
    grad_x = cv2.Sobel(thresh, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    grad_y = cv2.Sobel(thresh, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv2.imshow("normal", mit)
    cv2.imshow("sobel", grad)
    cv2.waitKey(0)


#Hvad er resultatet hvis du anvender Sobel kerner på hhv. RGB billede og gråtonebillede?
if opgave == 5:
    # størrelser
    scale = 0.3
    delta = 1
    ddepth = cv2.CV_16S

    mit = cv2.imread(r'C:\Users\mikip\Documents\Fag_Perception\Lektion_3\materialer\egetBillede.png')
    grey = cv2.imread(r'C:\Users\mikip\Documents\Fag_Perception\Lektion_3\materialer\egetBillede.png',
                      cv2.IMREAD_GRAYSCALE)
    blur = cv2.medianBlur(mit, 9)
    ret, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    # Gradient-X
    grad_x = cv2.Sobel(thresh, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    grad_y = cv2.Sobel(thresh, ddepth, 0, 1 , ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    blur = cv2.medianBlur(grey, 9)
    ret, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    # Gradient-X
    grad_x = cv2.Sobel(thresh, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    grad_y = cv2.Sobel(thresh, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad_grey = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv2.imshow("normal", mit)
    cv2.imshow("sobel", grad)
    cv2.imshow("sobel_grey", grad_grey)
    cv2.waitKey(0)

#Test Cannys edge detection
if opgave == 6:
    mit = cv2.imread(r'C:\Users\mikip\Documents\Fag_Perception\Lektion_3\materialer\egetBillede.png')
    grey = cv2.imread(r'C:\Users\mikip\Documents\Fag_Perception\Lektion_3\materialer\egetBillede.png',
                      cv2.IMREAD_GRAYSCALE)
    edge = cv2.Canny(mit, 0, 100)
    edge_grey = cv2.Canny(grey, 0, 100)

    cv2.imshow("normal", mit)
    cv2.imshow("canny", edge)
    cv2.imshow("canny_grey", edge_grey)
    cv2.waitKey(0)