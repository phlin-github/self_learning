import cv2 as cv
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = 'FingerImages'
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
pTime = 0

detector = htm.handDetector(min_detection_confidence=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.fingHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Four fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)

                # # Rock
                # if lmList[tipIds[2]][2] < lmList[tipIds[2] - 2][2] & lmList[tipIds[3]][2] < lmList[tipIds[3] - 2][2]:
                #     fingers.append(2)
            else:
                fingers.append(0)




        print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)
        # if fingers.append(2):
        #     h, w, c = overlayList[5].shape
        #     img[0:h, 0:w] = overlayList[5]
        # else:
        h, w, c = overlayList[totalFingers - 1].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1]

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(img, f'FPS:{int(fps)}', (500, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    cv.imshow('Image', img)
    cv.waitKey(1)
