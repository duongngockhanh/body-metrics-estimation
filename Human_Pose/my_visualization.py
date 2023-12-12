import cv2
import numpy as np

path = "quang2.jpg"

coors = np.array([[460, 480], [500, 560], [560, 630]])


img = cv2.imreadS(path)
for i in coors:
    img = cv2.circle(img, i, 5, (0, 0, 255), -1)

cv2.imwrite("myvisualization.jpg", img)
