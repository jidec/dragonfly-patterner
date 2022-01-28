from PIL import Image
from PIL import ImageOps
import numpy as np
import cv2

img_name = "4155610_198" #name of image minus extension to perform operations on
show = True

# open image and mask and convert to array
img = Image.open("images/" + img_name + ".jpg")
img = np.array(img, dtype=np.uint8)

mask = Image.open("images/" + img_name + "_mask" + ".tif")
mask = ImageOps.invert(mask) # invert mask so black represents background
mask = np.array(mask, dtype=np.uint8)

# apply mask to get masked img
masked_img = cv2.bitwise_and(img, img, mask=mask)

# create temp grayscale img, threshold against the black background, use to create a bounding rect
temp = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
th = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
coords = cv2.findNonZero(th)
x, y, w, h = cv2.boundingRect(coords)

# narrow image to bounding rect
masked_img = masked_img[y:y + h, x:x + w]

# add alpha channel, convert remaining black background to alpha
masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2BGRA)
alpha = masked_img[:, :, 3]
alpha[np.all(masked_img[:, :, 0:3] == (0, 0, 0), 2)] = 0

# convert final masked image to RGBA, write it, and show it
masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGRA2RGBA)
cv2.imwrite("segs/4155610_198_masked.png", masked_img)
if(show):
    cv2.imshow('Masked', masked_img)
    cv2.waitKey(0)

# load in masked_img and convert to array
img = Image.open("images/" + img_name + "_masked.png")
img = np.array(img, dtype=np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# reshape img for clustering
vals = img.reshape((-1,3))
vals = np.float32(vals)

# define criteria, number of clusters and apply kmeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv2.kmeans(vals,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# convert back to an image
center = np.uint8(center)
res = center[label.flatten()]
img = res.reshape((img.shape))

# convert from BGR to BGRA
img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

# get alpha channel and set alpha to 0 where RGB = 0 (pixels are black)
alpha = img[:, :, 3]
alpha[np.all(img[:, :, 0:3] == (0, 0, 0), 2)] = 0

if(show):
    cv2.imshow('Clustered',img)
    cv2.waitKey(0)

#plt.hist(res2.ravel(), 5, [0, 3]);
#plt.show()

# write boxed clustered image
cv2.imwrite("images/" + img_name + "_clust.png", img)
cv2.destroyAllWindows()