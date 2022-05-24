import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
facedection = mpFaceDetection.FaceDetection(0.75)
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = facedection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection)
            # print(detection.score)
            print(detection.location_data.relative_bounding_box)
            # mpDraw.draw_detection(img, detection)
            bboxC = detection.location_data.relative_bounding_box
            iheight, iwidth, ichannels = img.shape
            bbox = int(bboxC.xmin*iwidth),int(bboxC.ymin*iheight),int(bboxC.width*iwidth), int(bboxC.height*iheight)
            cv.rectangle(img, bbox, (255, 0, 255),2)
            cv.putText(img, f'{int(detection.score[0]*100)}% ', (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(img, f'FPS: {int(fps)}', (10,50), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv.imshow('Image', img)
    cv.waitKey(1)