import cv2 as cv
import mediapipe as mp
import time

class PoseDetector():
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.mode = static_image_mode
        self.complexity = model_complexity
        self.smooth = smooth_landmarks
        self.minCon = min_detection_confidence
        self.trackCon = min_tracking_confidence

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth, self.minCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    def getPosition(self, img, draw = True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                height, width, channels = img.shape
                x_center, y_center = int(lm.x*width), int(lm.y*height)
                lmList.append([id, x_center, y_center])
                if draw:
                    cv.circle(img, (x_center, y_center), 5, (255,0, 255), cv.FILLED)
        return lmList




def main():
    cap = cv.VideoCapture('Videos/6.mp4')
    pTime = 0
    detector = PoseDetector()
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

if __name__ == '__main__':
    main()