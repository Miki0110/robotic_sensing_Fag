import cv2

# Open picture
cool_rabbit = cv2.imread("wabbit.png", cv2.IMREAD_GRAYSCALE)

# Display picture
cv2.imshow("wabbit", cool_rabbit)
cv2.waitKey(0)