import cv2
from skimage.metrics import structural_similarity
import numpy as np

#mse between center is a good measure of symmetry
img = cv2.imread("E:\dragonfly-patterner\data\patterns\INAT-9376231-3_pattern.png")
img = cv2.imread("E:/dragonfly-patterner\data\patterns/aesh_grouped5000/INAT-4227056-1_pattern.png") # blue aeshnid nice
#img = cv2.imread("E:/dragonfly-patterner\data\patterns/aesh_grouped5000/INAT-808597-1_pattern.png") # asym blue aesh
img = cv2.imread("E:/dragonfly-patterner\data\patterns/aesh_grouped5000/INAT-910198-1_pattern.png") # pretty sym brown aesh
img = cv2.imread("E:/dragonfly-patterner\data\patterns/aesh_grouped5000/INAT-4266740-2_pattern.png") # simple blue aesh sym
img = cv2.imread("E:/dragonfly-patterner\data\patterns/aesh_grouped5000/INAT-31425449-1_pattern.png") # fucked image
img = cv2.imread("E:/dragonfly-patterner\data\patterns/aesh_grouped5000/INAT-31874313-3_pattern.png") # fat brown sym
img = cv2.imread("E:/dragonfly-patterner\data\patterns/gomph_grouped_5000/INAT-50359885-2_pattern.png") # super sym

cv2.imshow("i",img)
cv2.waitKey(0)
height = img.shape[0]
width = img.shape[1]

# Cut the image in half
width_cutoff = width // 2
img_left = img[:, :width_cutoff]
img_right = img[:, width_cutoff:]

cv2.imshow("i",img_left)
cv2.waitKey(0)

img_right = cv2.flip(img_right, 1)

cv2.imshow("i",img_right)
cv2.waitKey(0)

# resize right to left just in case
img_right = cv2.resize(img_right,(img_left.shape[1],img_left.shape[0]))
print(img_left.shape)
print(img_right.shape)
# Convert images to grayscale
img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

print(mse(img_left,img_right))

# Compute SSIM between two images
score, diff = structural_similarity(img_left, img_right, full=True)
print("Similarity Score: {:.3f}%".format(score * 100))