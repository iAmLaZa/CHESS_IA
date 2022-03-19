import numpy as np
import cv2


def countours(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, tresh = cv2.threshold(gray, 192, 210, cv2.THRESH_BINARY)
    countours, hearchy = cv2.findContours(tresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, countours, -1, (255, 0, 0), 5)
    cv2.imshow('image', img)
    cv2.waitKey(0)


countours('chessboard.png')




#goodFeaturesToTrack Detection Method 1

"""
img = cv2.imread('chessboard.png')
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
	x, y = corner.ravel()
	cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

cv2.imshow('goodFeaturesToTrack Detection Method 1',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""



#goodFeaturesToTrack Detection Method 2
"""
img = cv2.imread('chessboard.png')
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners= cv2.goodFeaturesToTrack(gray, 100, 0.01, 50)

for corner in corners:
    x,y= corner[0]
    x= int(x)
    y= int(y)
    cv2.rectangle(img, (x-5,y-5),(x+5,y+5),(255,0,0),-1)

cv2.imshow('goodFeaturesToTrack Detection Method 2',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""



#Harris Corner Detection Method

"""
image= cv2.imread('chessboard.png')
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray= np.float32(gray)

harris_corners= cv2.cornerHarris(gray, 3, 3, 0.05)
kernel= np.ones((5,5), np.uint8)
harris_corners= cv2.dilate(harris_corners, kernel, iterations= 2)
image[harris_corners > 0.025 * harris_corners.max()]= [255,0,0]
cv2.imshow( 'Harris Corner Detection Method',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""



