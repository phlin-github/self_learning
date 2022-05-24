import cv2 as cv
import time
import  Ch1_HandTrackingModule as htm

pTime = 0
cTime = 0
cap = cv.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.fingHands(img)
    lmList = detector.findPosition(img)
    # Print the coordinate of indicated point
    if len(lmList) != 0:
        print(lmList[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv.imshow('Image', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break