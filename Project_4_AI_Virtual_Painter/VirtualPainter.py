import cv2 as cv
import numpy as np
import time
import os
import HandTrackingModule as htm

###############################
brushThickness = 15
eraserThickness = 100
###############################
cap = cv.VideoCapture(0)
folderPath = 'Header'
myList = os.listdir(folderPath)
# print(myList)
overlayList = []

for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))

header = overlayList[0]
detector = htm.handDetector(min_detection_confidence=0.85)
drawColor = (0, 0, 0)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:

    # 1. Import image
    success, img = cap.read()
    img = cv.resize(img, (1280, 720))
    img = cv.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.fingHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)

        # Tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If Selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print('Selection Mode')
            # Checking for the click
            if y1 < 126:
                if 190 < x1 < 310:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 410 < x1 < 530:
                    header = overlayList[1]
                    drawColor = (0, 255, 255)
                elif 640 < x1 < 760:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 880 < x1 < 1000:
                    header = overlayList[3]
                    drawColor = (0, 0, 255)
                elif 1100 < x1 < 1240:
                    header = overlayList[4]
                    drawColor = (0, 0, 0)
            cv.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv.FILLED)

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED)
            print('Drawing Mode')
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (255, 255, 255):
                cv.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, imgCanvas)

    # Setting the header image
    img[0:126, 0:1280] = header
    # img = cv.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv.imshow('Image', img)
    cv.waitKey(1)
