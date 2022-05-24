import cv2 as cv
import numpy as np
import HandTrackingModule as htm
import time
import autopy

######################################
wCam, hCam = 640, 480
frameR = 100  # Frame reduction
smoothening = 5
######################################

cap = cv.VideoCapture(0)
cap.set(3, hCam)
cap.set(4, wCam)

plocX, plocY = 0, 0
clocX, clocY = 0, 0

pTime = 0
detector = htm.handDetector(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
wScr, hScr = autopy.screen.size()
print(wScr, hScr)

while True:
    # 1. Find hand landmarks
    success, img = cap.read()
    img = detector.fingHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only index finger :　Moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 6. Smoothen values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            # 7. Move mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            plocX, plocY = clocX, clocY

        # 8. Both index and middle fingers are up : Checking mode
        if fingers[1] == 1 and fingers[2] == 1:

            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)

            # 10. Click mouse if distance short
            if length < 45:
                cv.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv.FILLED)
                autopy.mouse.click()

    # 11. Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f'FPS:{int(fps)}', (20, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    # 12. Disply
    cv.imshow('Image', img)
    cv.waitKey(1)
