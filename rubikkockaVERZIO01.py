import cv2
import numpy as np

zold = cv2.imread("zold.jpg")
kek = cv2.imread("kek.jpg")
piros = cv2.imread("piros.jpg")
sarga = cv2.imread("sarga.jpg")
narancs = cv2.imread("narancs.jpg")
feher = cv2.imread("feher.jpg")
oldal = cv2.imread("kocka.png")

print("Rubik kocka oldalai importalodnak....")
zold = cv2.resize(zold, (1920,1080))
piros = cv2.resize(piros, (1920,1080))
kek = cv2.resize(kek, (1920,1080))
sarga = cv2.resize(sarga, (1920,1080))
narancs = cv2.resize(narancs, (1920,1080))
feher = cv2.resize(feher, (1920,1080))
print("Kep ujrameretezese: 1920x1080")
print("Kesz")
hsv = cv2.cvtColor(zold, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(piros, cv2.COLOR_BGR2HSV)
hsv3 = cv2.cvtColor(kek, cv2.COLOR_BGR2HSV)
hsv4 = cv2.cvtColor(sarga, cv2.COLOR_BGR2HSV)
hsv5 = cv2.cvtColor(narancs, cv2.COLOR_BGR2HSV)
hsv6 = cv2.cvtColor(feher, cv2.COLOR_BGR2HSV)
szintomb = [0,0,0]
lower_green = np.array([25, 189, 118])
upper_green = np.array([95, 255, 198])
lower_red = np.array([25, 189, 118])
upper_red = np.array([95, 255, 198])
lower_blue = np.array([25, 189, 118])
upper_blue = np.array([95, 255, 198])
lower_yellow = np.array([25, 189, 118])
upper_yellow = np.array([95, 255, 198])
lower_orange = np.array([25, 189, 118])
upper_orange = np.array([95, 255, 198])
lower_white = np.array([25, 189, 118])
upper_white = np.array([95, 255, 198])

mask = cv2.inRange(hsv, lower_green, upper_green)
mask1 = cv2.inRange(hsv, lower_green, upper_green)
mask2 = cv2.inRange(hsv, lower_green, upper_green)
mask3 = cv2.inRange(hsv, lower_green, upper_green)
mask4 = cv2.inRange(hsv, lower_green, upper_green)
mask5 = cv2.inRange(hsv, lower_green, upper_green)
mask6 = cv2.inRange(hsv, lower_green, upper_green)

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours2, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours3, _ = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours4, _ = cv2.findContours(mask4, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours5, _ = cv2.findContours(mask5, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours6, _ = cv2.findContours(mask6, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

def szin(avg,szintomb):
    print("belepett a fgvbe")
    print(avg)
    if avg[0] <= 200 and avg[1] >= 151 and avg[2] <= 100:
        print("zöld")
        szintomb = [0,128,0]
        return szintomb
    if avg[0] >= 150 and avg[1] <= 150 and avg[2] <= 100:
        print("kék")
        szintomb = [128, 0, 0]
        return szintomb
    if avg[0] <= 100 and avg[1] <= 100 and avg[2] >= 150:
        print("piros")
        szintomb = [0, 0, 128]
        return szintomb
    if avg[0] <= 50 and avg[1] >= 230 and avg[2] >= 230:
        print("sarga")
        szintomb = [255,255,0]
        return szintomb
    if avg[0] <= 50 and avg[1] >= 200 and avg[2] >= 200:
        print("narancs")
        szintomb = 	[255,165,0]
        return szintomb
    if avg[0] >240  and avg[1] >240 and avg[2] > 240:
        print("feher")
        szintomb = [255, 255, 255]
        return szintomb

def kirajzoltatas(row,col, i, szintomb):
    if i == 0:
        print("belépett eskü")
        for j in range(col[0], col[1]):
            for k in range(row[0],row[1]):
                oldal[j, k] = szintomb
    elif i == 1:
        print("belépett eskü")
        for j in range(col[2], col[3]):
            for k in range(row[0],row[1]):
                oldal[j, k] = szintomb
    elif i == 2:
        print("belépett eskü")
        for j in range(col[4], col[5]):
            for k in range(row[0],row[1]):
                oldal[j, k] = szintomb
    elif i == 3:
        print("belépett eskü")
        for j in range(col[0], col[1]):
            for k in range(row[2],row[3]):
                oldal[j, k] = szintomb
    elif i == 4:
        print("belépett eskü")
        for j in range(col[2], col[3]):
            for k in range(row[2],row[3]):
                oldal[j, k] = szintomb
    elif i == 5:
        print("belépett eskü")
        for j in range(col[4], col[5]):
            for k in range(row[2], row[3]):
                oldal[j, k] = szintomb
    elif i == 6:
        print("belépett eskü")
        for j in range(col[0], col[1]):
            for k in range(row[4], row[5]):
                oldal[j, k] = szintomb
    elif i == 7:
        print("belépett eskü")
        for j in range(col[2], col[3]):
            for k in range(row[4], row[5]):
                oldal[j, k] = szintomb
    elif i == 8:
        print("belépett eskü")
        for j in range(col[4], col[5]):
            for k in range(row[4], row[5]):
                oldal[j, k] = szintomb




i = 0
#kocka1 kirajzoltatas
row = np.array([0,220,234,450,464,683])
col = np.array([0,220,234,450,464,683])
for cnt in contours:
    if cv2.contourArea(cnt) >10000:
        print("area: ", cv2.contourArea(cnt))
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(zold,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.imshow('Kivagott kontur',zold[y:y+h,x:x+w])
        avg=np.array(cv2.mean(zold[y:y + h, x:x + w])).astype(np.uint8)
        print('Average color (BGR): ',avg)
        #cv2.imshow('Kivagott kontur1',zold[y:y+h,x:x+w])
        #ha zöld színű a kocka a képen
        if szin(avg,szintomb) == [0,128,0]:
            print("ebbe is belepett1")
            kirajzoltatas(row,col,i,[0,128,0])
        if szin(avg,szintomb) == [128,0,0]:
            print("ebbe is belepett2")
            kirajzoltatas(row,col,i,[128,0,0])
        if szin(avg,szintomb) == [0,0,128]:
            print("ebbe is belepett3")
            kirajzoltatas(row,col,i,[0,0,128])
        if szin(avg,szintomb) == [255,255,0]:
            print("ebbe is belepett4")
            kirajzoltatas(row,col,i,[255,255,0])
        if szin(avg,szintomb) == [255,165,0]:
            print("ebbe is belepett5")
            kirajzoltatas(row,col,i,[255,165,0])
        if szin(avg,szintomb) == [255, 255, 255]:
            print("ebbe is belepett6")
            kirajzoltatas(row,col,i,[255, 255, 255])
        print("i:", i)
        i = i+1
cv2.imshow("Kocka oldala színezettzold", oldal)
print("ITT VÉGZETT A ZÖLDDEL")
i = 0
#kek oldal
for cnt2 in contours2:
    if cv2.contourArea(cnt2) >10000:
        print("area: ", cv2.contourArea(cnt2))
        x,y,w,h = cv2.boundingRect(cnt2)
        cv2.rectangle(kek,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.imshow('Kivagott kontur',zold[y:y+h,x:x+w])
        avg=np.array(cv2.mean(kek[y:y + h, x:x + w])).astype(np.uint8)
        print('Average color (BGR): ',avg)
        #cv2.imshow('Kivagott kontur1',kek[y:y+h,x:x+w])
        #ha zöld színű a kocka a képen
        if szin(avg,szintomb) == [0,128,0]:
            print("ebbe is belepett1")
            kirajzoltatas(row,col,i,[0,128,0])
        if szin(avg,szintomb) == [128,0,0]:
            print("ebbe is belepett2")
            kirajzoltatas(row,col,i,[128,0,0])
        if szin(avg,szintomb) == [0,0,128]:
            print("ebbe is belepett3")
            kirajzoltatas(row,col,i,[0,0,128])
        if szin(avg,szintomb) == [255,255,0]:
            print("ebbe is belepett4")
            kirajzoltatas(row,col,i,[255,255,0])
        if szin(avg,szintomb) == [255,165,0]:
            print("ebbe is belepett5")
            kirajzoltatas(row,col,i,[255,165,0])
        if szin(avg,szintomb) == [255, 255, 255]:
            print("ebbe is belepett6")
            kirajzoltatas(row,col,i,[255, 255, 255])
        print("i:", i)
        i = i+1
cv2.imshow("Kocka oldala szinezettkek", oldal)



#contours2, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#Régi Kontúr rajzolás (bénábbik)
"""for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    area = cv2.contourArea(contour)
    if area>10000:
        cv2.drawContours(zold, [approx], 0, (0, 0, 0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        area = cv2.contourArea(contour)
        print(area)"""
i = 0
"""#kiterített kocka képen kontúrok keresése
for cont in contours2:
    terulet = cv2.contourArea(cont)
    print("terulet: ",terulet)
    i = i + 1
    print(i)
    if (cv2.contourArea(cont) >2000 and cv2.contourArea(cont) < 2500):
        x,y,w,h = cv2.boundingRect(cont)
        cv2.rectangle(kitkocka,(x,y),(x+w,y+h),(0,255,0),2)
        avg = np.array(cv2.mean(zold[y:y + h, x:x + w])).astype(np.uint8)
        #cv2.imshow('Kivagott kontur',kitkocka[y:y+h,x:x+w])
    if (cv2.contourArea(cont) >2500 and cv2.contourArea(cont) < 200000):
        approx = cv2.approxPolyDP(cont, 0.01 * cv2.arcLength(cont, True), True)
        cv2.drawContours(kitkocka, [approx], 0, (0, 0, 255),3)"""


avg1 = np.zeros(9)
print(avg1)
"""i = 0
#Megkeresi a beadott képen a középső kockának a színét, és az alapján fogja eldönteni, hogy a kiterített kockán
#hol kell majd színezni
for cnt in contours:
    if cv2.contourArea(cnt) >10000:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(zold,(x,y),(x+w,y+h),(0,255,0),2)
        avg1[i] = i+1
        i = i + 1
        print(avg1)
        if i == 5:
            avg = np.array(cv2.mean(zold[y:y + h, x:x + w])).astype(np.uint8)
            print('Average color (BGR): ', avg)
        #cv2.imshow('Kivagott kontur',zold[y:y+h,x:x+w])

print(avg1)
"""




cv2.imshow("Eredeti kep a zold oldalrol", zold)
cv2.imshow("Mask", mask)
cv2.imshow("HSV", hsv)
cv2.imshow("Kocka oldala színezett", oldal)
cv2.imshow("Eredeti kep a kek oldalrol", kek)
cv2.imshow("Mask2", mask2)
cv2.imshow("HSV2", hsv2)
cv2.imshow("Eredeti kep a piros oldalrol", piros)
cv2.imshow("Mask3", mask3)
cv2.imshow("HSV3", hsv3)

cv2.waitKey(0)
cv2.destroyAllWindows()