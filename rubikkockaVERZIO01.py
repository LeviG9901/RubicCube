import cv2
import numpy as np

zold = cv2.imread("zold.jpg")
kek = cv2.imread("kek.jpg")
piros = cv2.imread("piros.jpg")
sarga = cv2.imread("sarga.jpg")
narancs = cv2.imread("narancs.jpg")
feher = cv2.imread("feher.jpg")
kevert = cv2.imread("kevert.png")
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

szintomb = [0,0,0]

lower_green = np.array([25, 189, 118])
upper_green = np.array([95, 255, 198])
lower_blue = np.array([100,200,0])
upper_blue = np.array([140,255,255])
lower_red = np.array([66,65,170])
upper_red = np.array([200,200,240])
lower_yellow = np.array([30,100,100])
upper_yellow = np.array([180,230,230])
lower_orange = np.array([40,100,170])
upper_orange = np.array([138,180,230])
lower_white = np.array([140,120,70])
upper_white = np.array([255,255,255])


Green = [0,128,0]
Blue = [128,0,0]
Red = [0,0,128]

def hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv;
def zoldmaszk(hsv):
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return mask;
def kekmaszk(hsv):
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return mask;
def pirosmaszk(hsv):
    mask = cv2.inRange(hsv, lower_red, upper_red)
    return mask;
def sargamaszk(hsv):
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask;
def narancsmaszk(hsv):
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    return mask;
def fehermaszk(hsv):
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return mask;
def contourdraw(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
def szin(avg,szintomb):
    print("belepett a fgvbe")
    print(avg)
    if avg[0] <= 200 and avg[1] >= 151 and avg[2] <= 100:
        print("zöld")
        szintomb = [0,128,0]
        return szintomb
    elif avg[0] >= 150 and avg[1] <= 150 and avg[2] <= 100:
        print("kék")
        szintomb = [128, 0, 0]
        return szintomb
    elif avg[0] <= 120 and avg[1] <= 110 and avg[2] >= 150:
        print("piros")
        szintomb = [0, 0, 128]
        return szintomb
    elif avg[0] <= 50 and avg[1] >= 230 and avg[2] >= 230:
        print("sarga")
        szintomb = [255,255,0]
        return szintomb
    elif avg[0] <= 50 and avg[1] >= 200 and avg[2] >= 200:
        print("narancs")
        szintomb = 	[255,165,0]
        return szintomb
    elif avg[0] >240  and avg[1] >240 and avg[2] > 240:
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




"""
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
cv2.imshow("Eredeti kep a zold oldalrol", zold)
cv2.imshow("Mask", mask)
cv2.imshow("HSV", hsv)
print("ITT VÉGZETT A ZÖLDDEL")
"""
"""
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
cv2.imshow("Eredeti kep a kek oldalrol", kek)
cv2.imshow("Mask2", mask2)
cv2.imshow("HSV2", hsv2)
print("ITT VÉGZETT A KÉKKEL")
i = 0
#piros oldal
for cnt3 in contours3:
    if cv2.contourArea(cnt3) >10000:
        print("area: ", cv2.contourArea(cnt3))
        x,y,w,h = cv2.boundingRect(cnt3)
        cv2.rectangle(piros,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.imshow('Kivagott kontur',zold[y:y+h,x:x+w])
        avg=np.array(cv2.mean(piros[y:y + h, x:x + w])).astype(np.uint8)
        print('Average color (BGR): ',avg)
        #cv2.imshow('Kivagott kontur1',kek[y:y+h,x:x+w])
        #ha zöld színű a kocka a képen
        if szin(avg,szintomb) == Green:
            print("ebbe is belepett1")
            kirajzoltatas(row,col,i,[0,128,0])
        elif szin(avg,szintomb) == Blue:
            print("ebbe is belepett2")
            kirajzoltatas(row,col,i,[128,0,0])
        elif szin(avg,szintomb) == Red:
            print("ebbe is belepett3")
            kirajzoltatas(row,col,i,[0,0,128])
        elif szin(avg,szintomb) == [255,255,0]:
            print("ebbe is belepett4")
            kirajzoltatas(row,col,i,[255,255,0])
        elif szin(avg,szintomb) == [255,165,0]:
            print("ebbe is belepett5")
            kirajzoltatas(row,col,i,[255,165,0])
        elif szin(avg,szintomb) == [255, 255, 255]:
            print("ebbe is belepett6")
            kirajzoltatas(row,col,i,[255, 255, 255])
        print("i:", i)
        i = i+1
cv2.imshow("Kocka oldala szinezettpiros", oldal)
cv2.imshow("Eredeti kep a piros oldalrol", piros)
cv2.imshow("Mask3", mask3)
cv2.imshow("HSV3", hsv3)
print("ITT VÉGZETT A PIROSSAL")
i = 0
#piros oldal
for cnt4 in contours4:
    if cv2.contourArea(cnt4) >10000:
        print("area: ", cv2.contourArea(cnt4))
        x,y,w,h = cv2.boundingRect(cnt4)
        cv2.rectangle(kevert,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.imshow('Kivagott kontur',zold[y:y+h,x:x+w])
        avg=np.array(cv2.mean(kevert[y:y + h, x:x + w])).astype(np.uint8)
        print('Average color (BGR): ',avg)
        #cv2.imshow('Kivagott kontur1',kek[y:y+h,x:x+w])
        #ha zöld színű a kocka a képen
        if szin(avg,szintomb) == Green:
            print("ebbe is belepett1")
            kirajzoltatas(row,col,i,[0,128,0])
        elif szin(avg,szintomb) == Blue:
            print("ebbe is belepett2")
            kirajzoltatas(row,col,i,[128,0,0])
        elif szin(avg,szintomb) == Red:
            print("ebbe is belepett3")
            kirajzoltatas(row,col,i,[0,0,128])
        elif szin(avg,szintomb) == [255,255,0]:
            print("ebbe is belepett4")
            kirajzoltatas(row,col,i,[255,255,0])
        elif szin(avg,szintomb) == [255,165,0]:
            print("ebbe is belepett5")
            kirajzoltatas(row,col,i,[255,165,0])
        elif szin(avg,szintomb) == [255, 255, 255]:
            print("ebbe is belepett6")
            kirajzoltatas(row,col,i,[255, 255, 255])
        print("i:", i)
        i = i+1
cv2.imshow("Kocka oldala szinezettkevert", oldal)
cv2.imshow("Eredeti kep a kevert oldalrol", kevert)
cv2.imshow("Mask7", mask7)
cv2.imshow("HSV7", hsv7)
print("ITT VÉGZETT A KEVERTTEL")

#contours2, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#Régi Kontúr rajzolás (bénábbik)
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    area = cv2.contourArea(contour)
    if area>10000:
        cv2.drawContours(zold, [approx], 0, (0, 0, 0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        area = cv2.contourArea(contour)
        print(area)"""
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

cv2.waitKey(0)
cv2.destroyAllWindows()