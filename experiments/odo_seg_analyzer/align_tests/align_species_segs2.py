from PIL import Image
from PIL import ImageOps
import numpy as np
import cv2
import math

img1name = "4155610_198"
img2name = "56568289_564"
#img1name = "56568289_564"
show = True

# read images to align
img = cv2.imread("images/" + img1name + "_clust.png",cv2.IMREAD_GRAYSCALE)
#img2 = cv2.imread("images/" + img2name + "_clust.png",cv2.IMREAD_GRAYSCALE)

#img = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)  # Read input image as grayscale.

threshed = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]  # threshold (binarize) the image

# Apply closing for connecting the lines
threshed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, np.ones((1, 10)))

# Find contours
contours = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # [-2] indexing takes return value before last (due to OpenCV compatibility issues).

img2 = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)  # BGR image - used for drawing

angles = []  # List of line angles.

# Iterate the contours and fit a line for each contour
# Remark: consider ignoring small contours
for c in contours:
    vx, vy, cx, cy = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01) # Fit line
    w = img.shape[1]
    cv2.line(img2, (int(cx-vx*w), int(cy-vy*w)), (int(cx+vx*w), int(cy+vy*w)), (0, 255, 0))  # Draw the line for testing
    ang = (180/np.pi)*math.atan2(vy, vx) # Compute the angle of the line.
    angles.append(ang)

angles = np.array(angles)  # Convert angles to NumPy array.

# Remove outliers and
lo_val, up_val = np.percentile(angles, (40, 60))  # Get the value of lower and upper 40% of all angles (mean of only 10 angles)
mean_ang = np.mean(angles[np.where((angles >= lo_val) & (angles <= up_val))])

print(f'mean_ang = {mean_ang}')  # -0.2424

M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), mean_ang, 1)  # Get transformation matrix - for rotating by mean_ang

img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), cv2.INTER_CUBIC) # Rotate the image

# Display results
cv2.imshow('img2', img2)
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()