import cv2 as cv
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self,min_detection_confidence=0.5, model_selection=0):
        self.minDetectCon = min_detection_confidence
        self.modelSel = model_selection


        self.mpFaceDetection = mp.solutions.face_detection
        self.facedection = self.mpFaceDetection.FaceDetection(self.minDetectCon, self.modelSel)
        self.mpDraw = mp.solutions.drawing_utils

    def findFace(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.facedection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                iheight, iwidth, ichannels = img.shape
                bbox = int(bboxC.xmin*iwidth),int(bboxC.ymin*iheight),int(bboxC.width*iwidth), int(bboxC.height*iheight)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    self.fancyDraw(img, bbox)
                    cv.putText(img, f'{int(detection.score[0]*100)}% ', (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        return bboxs, img

    def fancyDraw(self, img, bbox, l = 30, t = 5, rt = 1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h

        cv.rectangle(img, bbox, (255, 0, 255), rt)
        # Top Left x, y
        cv.line(img, (x, y), (x+l, y), (255, 0, 255), t)
        cv.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top Right x1, y
        cv.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # Bottom Left x, y1
        cv.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right x1, y1
        cv.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img
def main():
    cap = cv.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        bboxs, img = detector.findFace(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, f'FPS: {int(fps)}', (10, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

        cv.imshow('Image', img)
        cv.waitKey(1)
if __name__ == '__main__':
    main()