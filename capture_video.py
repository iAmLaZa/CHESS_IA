import cv2
import numpy as np
def countours(gray, img):
    ret, tresh = cv2.threshold(gray, 192, 210, cv2.THRESH_BINARY)
    countours, hearchy = cv2.findContours(tresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, countours, -1, (255, 0, 0), 5)



cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True :
    ret ,frame=cap.read()
    frame = cv2.resize(frame, (526, 392))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    countours(gray, frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1)==ord('q') :
        break

cap.release()
cv2.destroyAllWindows()