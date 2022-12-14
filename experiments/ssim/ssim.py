from skimage.metrics import structural_similarity
import cv2
import numpy as np

#first = cv2.imread("E:\dragonfly-patterner\data\patterns\INAT-9376231-3_pattern.png") # bit asym aeshnid
#img2 = cv2.imread("E:\dragonfly-patterner\data\patterns\INAT-118990275-4_pattern.png") # black abdomen
img2 = cv2.imread("E:\dragonfly-patterner\data\patterns\INAT-104899633-1_pattern.png") # bright dorsal gomphid sym
#img = cv2.imread("E:\dragonfly-patterner\data\patterns\gomph_grouped_5000\INAT-130715589-1_pattern.png") # dorsal gomphid
#img2 = cv2.imread("E:\dragonfly-patterner\data\patterns\INAT-4227056-1_pattern.png") # sym dorsal blue aesh
img = cv2.imread("E:\dragonfly-patterner\data\patterns\gomph_grouped_5000\INAT-130715589-1_pattern.png") # dorsal gomphid
img2 = cv2.imread("E:\dragonfly-patterner\data\patterns\gomph_grouped_5000\INAT-28006702-1_pattern.png") # fucked gomphid
#img2 = cv2.imread("E:\dragonfly-patterner\data\patterns\INAT-4227056-1_pattern.png") # sym dorsal blue aesh
#img2 = cv2.imread("E:\dragonfly-patterner\data\patterns\INAT-4227056-1_pattern.png") # sym dorsal blue aesh
#img2 = cv2.imread("E:\dragonfly-patterner\data\patterns\INAT-4227056-1_pattern.png") # sym dorsal blue aesh

img2 = cv2.resize(img2, (img.shape[1],img.shape[0]))


print(img.shape)
print(img2.shape)
# Convert images to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Compute SSIM between two images
score, diff = structural_similarity(img_gray, img2_gray, full=True)
print("Similarity Score: {:.3f}%".format(score * 100))

def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

print(mse(img_gray,img2_gray))
# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type so we must convert the array
# to 8-bit unsigned integers in the range [0,255] before we can use it with OpenCV
diff = (diff * 255).astype("uint8")

# Threshold the difference image, followed by finding contours to
# obtain the regions that differ between the two images
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

# Highlight differences
mask = np.zeros(img.shape, dtype='uint8')
filled = img2.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area > 100:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, (0,255,0), -1)
        cv2.drawContours(filled, [c], 0, (0,255,0), -1)

cv2.imshow('img', img)
cv2.imshow('img2', img2)
cv2.imshow('diff', diff)
cv2.imshow('mask', mask)
cv2.imshow('filled', filled)
cv2.waitKey()