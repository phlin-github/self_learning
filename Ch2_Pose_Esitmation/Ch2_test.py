import cv2 as cv
import mediapipe as mp
import time
import PoseEstimationModule as pem

cap = cv.VideoCapture('Videos/4.mp4')
pTime = 0
detector = pem.PoseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.getPosition(img)
    print(lmList)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv.imshow('Image', img)
    cv.waitKey(1)