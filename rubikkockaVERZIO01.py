import cv2
import numpy as np

zold = cv2.imread("zold.jpg")
kocka = cv2.imread("kocka.png")
print("Rubik kocka zold oldala importalodik....")
zold = cv2.resize(zold, (1920,1080))
print("Kep ujrameretezese: 1920x1080")
print("Kesz")
hsv = cv2.cvtColor(zold, cv2.COLOR_BGR2HSV)

lower_green = np.array([25, 189, 118])
upper_green = np.array([95, 255, 198])

mask = cv2.inRange(hsv, lower_green, upper_green)

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    area = cv2.contourArea(contour)
    if area>10000:
        cv2.drawContours(zold, [approx], 0, (0, 0, 0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        area = cv2.contourArea(contour)
        print(area)
for cnt in contours:
    if cv2.contourArea(cnt) >10000:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(zold,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.imshow('Kivagott kontur',zold[y:y+h,x:x+w])
        avg=np.array(cv2.mean(zold[y:y + h, x:x + w])).astype(np.uint8)
        print('Average color (BGR): ',avg)
        cv2.waitKey(0)

x = 221
y = 221
for i in range(x):
      for j in range(y):
        kocka[i,j] = avg[:3]


cv2.imshow("Eredeti kep a zold oldalrol", zold)
cv2.imshow("Mask", mask)
cv2.imshow("HSV", hsv)
cv2.imshow("Kocka", kocka)
cv2.waitKey(0)
cv2.destroyAllWindows()