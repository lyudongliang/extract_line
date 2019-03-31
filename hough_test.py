import cv2
import numpy as np
import time
import os


type = 'walk'
# type = 'road'
# type = 'niu'
# type = 'home'
# type = 'table'
# type = 'window'
# type = 'office'

picture_file = type + '_corner.jpg'
picture_directory = 'D:\data\picture'

src_picture = os.path.join(picture_directory, picture_file)

img = cv2.imread(src_picture)

time0 = time.time()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (9, 9), 0)
edges = cv2.Canny(gray, 30, 100, apertureSize=3)
# edges = cv2.Sobel(gray, 30, 100, apertureSize=3)
edge_file = type + '_corner_edge.jpg'

cv2.imwrite(os.path.join('picture', edge_file), edges)

lines = cv2.HoughLinesP(edges, 1, 0.5 * np.pi/180, 200, minLineLength=20, maxLineGap=50)

print('time', time.time() - time0)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

output_dirctory = os.path.join('picture', picture_file)
cv2.imwrite(output_dirctory, img)

print('finish')


