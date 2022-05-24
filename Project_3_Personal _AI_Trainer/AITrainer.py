import cv2 as cv
import time
import numpy as np
import PoseEstimationModule as pem

cap = cv.VideoCapture(0)
detector = pem.PoseDetector(min_detection_confidence=0.75, min_tracking_confidence=0.75)
pTime = 0

count = 0
dir = 0

while True:
    success, img = cap.read()
    img = cv.resize(img, (640, 480))
    img = detector.findPose(img, False)
    lmList = detector.getPosition(img, False)
    # print(lmList)
    if len(lmList) != 0:
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (40, 170), (0, 100))
        bar = np.interp(angle, (40,170), (300, 100))
        # print(angle, per)

        # Check for the dumbbell curls
        if per == 100:
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            if dir == 1:
                count += 0.5
                dir = 0
        # print(count)

        cv.rectangle(img, (50, 100), (70, 300), (255, 255, 255), 2)
        cv.rectangle(img, (50, int(bar)), (70,300), (255,255,255), cv.FILLED)
        cv.putText(img, f'{int(per)} %', (20, 90), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

        cv.rectangle(img, (0, 380), (100,480), (0,255,255), cv.FILLED)
        cv.putText(img, f'{int(count)}', (10, 455), cv.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f'FPS:{int(fps)}', (5, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    cv.imshow('Image', img)
    cv.waitKey(1)
