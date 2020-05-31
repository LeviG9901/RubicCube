import cv2
import numpy as np

"""
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
"""

def kepvalasztas(r):
    if r == 1:
        zold = cv2.imread("zold.jpg")
        zold = cv2.resize(zold, (1920,1080))
        return zold
    elif r == 2:
        kek = cv2.imread("kek.jpg")
        kek = cv2.resize(kek, (1920,1080))
        return kek
    elif r == 3:
        piros = cv2.imread("piros.jpg")
        piros = cv2.resize(piros, (1920,1080))
        return piros 
    elif r == 4:
        sarga = cv2.imread("sarga.jpg")
        sarga = cv2.resize(sarga, (1920,1080))
        return sarga
    elif r == 5:
        narancs = cv2.imread("narancs.jpg")
        narancs = cv2.resize(narancs, (1920,1080))
        return narancs
    elif r == 6:
        feher = cv2.imread("feher.jpg")
        feher = cv2.resize(feher, (1920,1080))
        return feher

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





lowup =np.array ([[42,55,60],
                 [93, 255, 255],
                 [100,151,200],
                 [106,255,255],
                 [171,130,0],
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




for r in range (1,7):
    print("r elejen:", r)
    hsv = cv2.cvtColor(kepvalasztas(r), cv2.COLOR_BGR2HSV)
    cszintomb=np.array([])
    j = 0
    for i in range(0, 12 ,2):
       k = 0
       mask = cv2.inRange(hsv,lowup[i],lowup[i+1])
       print("i:",i)
       print("jelejen:",j)
       contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
       for cnt in contours:
           if cv2.contourArea(cnt) > 10000:
               x, y, w, h = cv2.boundingRect(cnt)
               # ITT KELL MÓDOSÍTANI rectangle(IDE ÍRNI A KÉP NEVÉT)!!!!!!!!!!!!!!!!!!!
               cv2.rectangle(kepvalasztas(r), (x, y), (x + w, y + h), (0, 255, 0), 2)
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
               k = k+1
               print("k:", k)
               
       j = j + 1
       print("jvegen:",j)

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

    o=1
    for z in range(9):
        for q in range(3):
            if (szin(asd[z][q])=="G"):
                print('G',end='')
            elif (szin(asd[z][q])=="B"):
                print('B',end='')
            elif (szin(asd[z][q])=="R"):
                print('R',end='')
            elif (szin(asd[z][q])=="Y"):
                print('Y',end='')
            elif (szin(asd[z][q])=="O"):
                print('O',end='')
            elif (szin(asd[z][q])=="W"):
                print('W',end='')
        if (o%3)==0:
            print("\n")
        o=o+1
#print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


cv2.waitKey(0)
cv2.destroyAllWindows()