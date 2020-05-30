import cv2
import numpy as np

zold = cv2.imread("zold.jpg")
kek = cv2.imread("kek.jpg")
piros = cv2.imread("piros.jpg")
sarga = cv2.imread("sarga.jpg")
narancs = cv2.imread("narancs.jpg")
feher = cv2.imread("feher.jpg")
zavart = cv2.imread("zavarotenyezok01.jpg")
zavart2 = cv2.imread("zavarotenyezok02.jpg")




print("Rubik kocka oldalai importalodnak....")
zold = cv2.resize(zold, (1920,1080))
piros = cv2.resize(piros, (1920,1080))
kek = cv2.resize(kek, (1920,1080))
sarga = cv2.resize(sarga, (1920,1080))
narancs = cv2.resize(narancs, (1920,1080))
feher = cv2.resize(feher, (1920,1080))
zavart = cv2.resize(feher, (1920,1080))
zavart2 = cv2.resize(feher, (1920,1080))
print("Kep ujrameretezese: 1920x1080")
print("Kesz")


def kepvalasztas(r):
    if r == 1:
        return cv2.imread("zold.jpg")
    elif r == 2:
        print("kékbe belépett ám")
        return cv2.imread("kek.jpg")
    elif r == 3:
        return cv2.imread("piros.jpg")
    elif r == 4:
        return cv2.imread("sarga.jpg")
    elif r == 5:
        return cv2.imread("narancs.jpg")
    elif r == 6:
        return cv2.imread("feher.jpg")

def szin(j):
    if j == 0:
        return "G"
    if j == 1:
        return "B"
    if j == 2:
        return "R"
    if j == 3:
        return "Y"
    if j == 4:
        return "O"
    if j == 5:
        return "W"
"""
def szin(avg,szintomb):
    print("belepett a fgvbe")
    print("avg",avg)
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
"""
"""
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

szintomb = [0,0,0]
Green = [0,128,0]
Blue = [128,0,0]
Red = [0,0,128]
Yellow = []

lower_green = np.array([40, 90, 20])
upper_green = np.array([210, 245, 200])
lower_blue = np.array([72,199,130])
upper_blue = np.array([137,255,255])
lower_red = np.array([20,10,140])
upper_red = np.array([160,160,248])
lower_yellow = np.array([30,100,100])
upper_yellow = np.array([180,230,230])
lower_orange = np.array([40,100,170])
upper_orange = np.array([138,180,230])
lower_white = np.array([140,120,70])
upper_white = np.array([255,255,255])

"""
def függvény(lowup):
    for i in range(0, 6, 2):
        mask = cv2.inRange(hsv, lowup[i], lowup[i + 1])
        return mask

"""



lowup =np.array ([[42,55,60],
                 [93, 255, 255],
                 [100,151,200],
                 [106,255,255],
                 [169,128,0],
                 [255,255,255],
                 [21,121,144],
                 [255,255,196],
                 [0,86,169],
                 [63,255,255],
                 [62,0,130],
                 [255,255,255]])
print(lowup[0])

i = 0
j = 0 #ez adja meg a maszk színt, éppen mit maszkol ki
k = 0

cszintomb=np.array([])


#ITT KELL MÓDOSÍTANI cvtCOLOR(IDE ÍRNI A KÉP NEVÉT)!!!!!!!!!!!!!!!!!!!
hsv = cv2.cvtColor(zavart, cv2.COLOR_BGR2HSV)
for i in range(0, 12 ,2):
       mask = cv2.inRange(hsv,lowup[i],lowup[i+1])
       print("i:",i)
       print("j:",j)
       contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
       for cnt in contours:
           if cv2.contourArea(cnt) > 10000:
               x, y, w, h = cv2.boundingRect(cnt)
               cv2.rectangle(kevert, (x, y), (x + w, y + h), (0, 255, 0), 2)
               #M = cv2.moments(cnt)
               #print("Momment:",M)
               #cx =(M['m10'] / M['m00'])
               #print("cx:",cx)
               #cy =(M['m01'] / M['m00'])
               #print("cy:", cy)
               #center = np.array([cx,cy])
               #print("momentscenter:", center)
               circles = [cv2.minEnclosingCircle(cnt)]
               M2 = cv2.moments(cnt)
               #print("Momment2:",M2)
               center2 = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))
               #print("circlecenter:", center2)
               cx= int(center2[0])
               cy = int(center2[1])
               cxcyj=np.empty(3)
               cxcyj[0] = cx
               cxcyj[1] = cy
               cxcyj[2] = j
               #print("cxcyj:", cxcyj)
               cszintomb = np.append(cszintomb,cxcyj)
               print("cszintomb:", cszintomb)
               avg=np.array(cv2.mean(kevert[y:y + h, x:x + w])).astype(np.uint8)
               #print('Average color (BGR): ', avg)
               k = k+1
               print("k:", k)
       j = j + 1

print("cszintomb a vegen:", cszintomb)
#segedtömb segítségével csinálok ebből egy 9x3 mátrixot
segedt = np.ndarray(shape=(9),dtype=[('x','f4'),('y','f4'),('c','f4')])
print("segedt", segedt)
l = 0
for t in range(0,9):
    for u in range(0,3):
        segedt[t][u] = cszintomb[l]
        l = l+1

print("segedt:", segedt)
#y koordináta szerint sorbarendezem (axis=0)
asd = sorted(segedt, key=lambda segedt_entry: (segedt_entry[1],segedt_entry[0]))
print("sorted:", asd)
"""
sortomb = np.ndarray(shape=(3,3))
for i in range (3):
    for j in range (3):
        sortomb[i][j] = segedt[i][j]
print("sortomb: ", sortomb)
"""


o=1
for i in range(9):

    for j in range(3):
        if (szin(asd[i][j])=="G"):
            print('G',end='')
        elif (szin(asd[i][j])=="B"):
            print('B',end='')
        elif (szin(asd[i][j])=="R"):
            print('R',end='')
        elif (szin(asd[i][j])=="Y"):
            print('Y',end='')
        elif (szin(asd[i][j])=="O"):
            print('O',end='')
        elif (szin(asd[i][j])=="W"):
            print('W',end='')
    if (o%3)==0:
        print("\n")
    o=o+1
#print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
"""
i = 0
j = 0 #ez adja meg a maszk színt, éppen mit maszkol ki
k = 0
for i in range(0, 12 ,2):
       mask = cv2.inRange(hsv2,lowup[i],lowup[i+1])
       cv2.imshow("mask", mask)
       print("i:",i)
       contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
       for cnt in contours:
           if cv2.contourArea(cnt) > 10000:

               x, y, w, h = cv2.boundingRect(cnt)
               cv2.rectangle(kevert, (x, y), (x + w, y + h), (0, 255, 0), 2)
               M = cv2.moments(cnt)
               print("Momment:",M)
               cx =(M['m10'] / M['m00'])
               print("cx:",cx)
               cy =(M['m01'] / M['m00'])
               print("cy:", cy)
               center = np.array([cx,cy])
               print("momentscenter:", center)
               circles = [cv2.minEnclosingCircle(cnt)]
               M2 = cv2.moments(cnt)
               print("Momment2:",M2)
               center2 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
               print("circlecenter:", center2)
               cx= int(center2[0])
               cy = int(center2[1])
               cxcyj=np.empty(3)
               cxcyj[0] = cx
               cxcyj[1] = cy
               cxcyj[2] = j
               print("cxcyj:", cxcyj)
               cszintomb = np.append(cszintomb,cxcyj)
               print("cszintomb:", cszintomb)
               avg=np.array(cv2.mean(kevert[y:y + h, x:x + w])).astype(np.uint8)
               print('Average color (BGR): ', avg)
               k = k+1
               print("k:", k)
       j = j + 1

print("cszintomb a vegen:", cszintomb)
#segedtömb segítségével csinálok ebből egy 9x3 mátrixot
segedt = np.ndarray(shape=(9),dtype=[('x','f4'),('y','f4'),('c','f4')])
print("segedt", segedt)
l = 0
for t in range(0,9):
    for u in range(0,3):
        segedt[t][u] = cszintomb[l]
        l = l+1

print("segedt:", segedt)
#y koordináta szerint sorbarendezem (axis=0)
asd = sorted(segedt, key=lambda segedt_entry: (segedt_entry[1],segedt_entry[0]))
print("sorted:", asd)
"""
""""
sortomb = np.ndarray(shape=(3,3))
for i in range (3):
    for j in range (3):
        sortomb[i][j] = segedt[i][j]
print("sortomb: ", sortomb)
"""

"""
o=1
for i in range(9):

    for j in range(3):
        if (szin(asd[i][j])=="G"):
            print('G',end='')
        elif (szin(asd[i][j])=="B"):
            print('B',end='')
        elif (szin(asd[i][j])=="R"):
            print('R',end='')
        elif (szin(asd[i][j])=="Y"):
            print('Y',end='')
        elif (szin(asd[i][j])=="O"):
            print('O',end='')
        elif (szin(asd[i][j])=="W"):
            print('W',end='')
    if (o%3)==0:
        print("\n")
    o=o+1
"""
#Kevert kocka oldalon (zöld,kék,piros színekkel) felismerés


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