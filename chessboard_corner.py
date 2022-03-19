import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

IMAGE_PATH = "3d1.jpg"
def convert_cv_hsv(color):
  color = np.array(color, dtype=np.float32)
  coeffs = np.array([179/360, 2.55, 2.55], dtype=np.float32)
  return np.uint8(color*coeffs)


img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (800,500))
low_blue = convert_cv_hsv((200, 70, 60)) # low_hue, low_sat, low_val
high_blue = convert_cv_hsv((260, 100, 85)) # high_hue, high_sat, high_val

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_img, low_blue, high_blue)
only_corners = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('image',only_corners)


# mask all the black pixels
mask = (only_corners!=0).all(axis=-1).flatten()

# find the index of non-masked pixels
index = np.array(list(np.ndindex(img.shape[:-1])))

Z = np.float32(index[mask])

kmeans = KMeans(n_clusters=4)
kmeans.fit(Z)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)
kmeans.fit(Z)
centers = np.uint16(kmeans.cluster_centers_)
plt.scatter(Z[:,0], Z[:,1])
plt.scatter(centers[:,0], centers[:,1])
plt.show()

centers = np.uint16(kmeans.cluster_centers_)

c_img = np.copy(img)

for center in centers:
  cv2.circle(c_img, tuple(center[::-1]), 5, (0,0,255), -1)
  cv2.circle(c_img, tuple(center[::-1]), 5, (0,0,255), -1)
  cv2.circle(c_img, tuple(center[::-1]), 5, (0,0,255), -1)
  cv2.circle(c_img, tuple(center[::-1]), 5, (0,0,255), -1)


cv2.imshow("c_img",c_img)


df = pd.DataFrame(centers, columns = ["x","y"])

min_x = df.sort_values('x').to_numpy()[:2]
min_y = df.sort_values('y').to_numpy()[:2]
max_x = df.sort_values('x', ascending=False).to_numpy()[:2]
max_y = df.sort_values('y', ascending=False).to_numpy()[:2]

# this will help us identify which corner is which
for center in centers:
  if (center in min_x) and (center in min_y):
    top_left = center[::-1]
  elif (center in min_x) and (center in max_y):
    top_right = center[::-1]
  elif (center in max_x) and (center in min_y):
    bottom_left = center[::-1]
  elif (center in max_x) and (center in max_y):
    bottom_right = center[::-1]

result = np.copy(img)

vertices = np.array([top_left, top_right, bottom_right, bottom_left], np.int32).reshape((-1,1,2))
cv2.polylines(result,[vertices],True,(0,255,0))

cv2.imshow("result",result)



mask = np.zeros(img.shape, np.uint8)
mask = np.uint8(np.where(mask==0, 0, 1))
mask = cv2.fillPoly(mask, pts=[vertices], color=(255, 255, 255))

mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
mask = np.where(mask==0, False, True)

index = np.ndindex(mask.shape)
masked_img=np.copy(img)
for idx in index:
  if not mask[idx]:masked_img[idx]=np.zeros(3)
cv2.imwrite('masked.png',masked_img)
cv2.imshow("masked_img",masked_img)



cv2.waitKey(0)