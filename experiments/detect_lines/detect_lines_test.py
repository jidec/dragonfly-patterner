import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread("test_wing2.png",cv2.IMREAD_COLOR)
# convert to grayscale
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# perform edge detection
edges = cv2.Canny(grayscale, 75, 200) #30

# detect lines in the image using hough lines technique
lines = cv2.HoughLinesP(image=edges, rho=2, theta=np.pi/180, threshold=80, lines=np.array([]), minLineLength=10, maxLineGap=10)
# iterate over the output lines and draw them
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        #cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 3)
# show images
cv2.imshow("image", image)
cv2.imshow("edges", edges)
cv2.waitKey(0)