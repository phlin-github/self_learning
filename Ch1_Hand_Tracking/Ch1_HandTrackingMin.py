import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                height, width, channels = img.shape
                x_center, y_center = int(lm.x*width), int(lm.y*height)
                print(id, x_center, y_center)
                if id == 0:
                    cv.circle(img, (x_center,y_center), 10, (225, 0, 255), cv.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)),(10,70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


    cv.imshow('Image', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
