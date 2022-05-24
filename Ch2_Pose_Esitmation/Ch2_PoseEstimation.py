import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture('Videos/6.mp4')
pTime = 0

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            height, width, channels = img.shape
            x_center, y_center = int(lm.x*width), int(lm.y*height)
            print(id, x_center, y_center)
            cv.circle(img, (x_center, y_center), 5, (255,0, 255), cv.FILLED)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv.imshow('Image', img)
    cv.waitKey(1)
