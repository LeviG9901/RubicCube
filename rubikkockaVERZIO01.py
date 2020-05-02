import cv2
import numpy as np

zold = cv2.imread("zold.jpg")
print("Rubik kocka zold oldala importalodik...")
zold = cv2.resize(zold, (1920,1080))
print("Kep ujrameretezese: 1920x1080")
print("Kesz")
hsv = cv2.cvtColor(zold, cv2.COLOR_BGR2HSV)

lower_green = np.array([0, 200, 0])
upper_green = np.array([255, 255, 255])

mask = cv2.inRange(hsv, lower_green, upper_green)
 

cv2.imshow("Eredeti kep a zold oldalrol", zold)
cv2.imshow("Mask", mask)
cv2.imshow("HSV", hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()