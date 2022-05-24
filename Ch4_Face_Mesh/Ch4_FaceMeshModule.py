import cv2 as cv
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.staticmode = static_image_mode
        self.maxNumfaces = max_num_faces
        self.minDetectCon = min_detection_confidence
        self.minTrackCon = min_tracking_confidence

        self.mpFaceMesh = mp.solutions.face_mesh
        self.Facemesh = self.mpFaceMesh.FaceMesh(self.staticmode, self.maxNumfaces, self.minDetectCon, self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def getPosition(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.Facemesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, facelms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(facelms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    # Add number to each id
                    # cv.putText(img, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                    face.append([x,y])
                faces.append(face)
        return img, faces

def main():
    cap = cv.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.getPosition(img)
        if len(faces) != 0:
            print(faces[0])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, f'FPS:{int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

        cv.imshow('Image', img)
        cv.waitKey(1)

if __name__ == '__main__':
    main()