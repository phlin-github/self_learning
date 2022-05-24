import cv2
import time

 

cap = cv2.VideoCapture(0)

#Check whether user selected camera is opened successfully.

if not (cap.isOpened()):
    print('Could not open video device')
else:
    print('Video device opened')

 

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

 

while(True):
  ret, frame = cap.read()

  # 將圖片轉為灰階
  #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  #cv2.imwrite('test.jpg', frame)
 

  cv2.imshow('frame', frame )

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

 

cap.release()
cv2.destroyAllWindows()