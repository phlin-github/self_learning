import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)
pTime = 0

mpFaceMesh = mp.solutions.face_mesh
Facemesh = mpFaceMesh.FaceMesh(max_num_faces=2)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = Facemesh.process(imgRGB)
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, facelms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)
            for id, lm in enumerate(facelms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                print(id, x, y)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, f'FPS:{int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv.imshow('Image', img)
    cv.waitKey(1)