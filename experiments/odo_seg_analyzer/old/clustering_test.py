from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
from PIL import Image
import numpy as np
import cv2
from sklearn.neighbors import NearestCentroid
from extractSegment import extractSegment

name = "7976280_543"
name = "56568289_564"
extracted = extractSegment(img_name = name, img_dir = "images", mask_dir="images",show=False) 

colorDiscretizeIndvContours(rgba_img_and_mask=extracted,img_name=name)

img = Image.open("images/" + "4155610_198_masked" + ".png")
img = np.array(img, dtype=np.uint8)
#convert img to grey
img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#set a thresh
thresh = 100
#get threshold image
ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
#find contours
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#create an empty image for contours
img_contours = np.zeros(img.shape)
# draw the contours on the empty image
cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)
cv2.imshow('Masked', img_contours)


cv2.waitKey(0)
#cluster_model = "kmeans"
#cluster_model = "optics"
#cluster_model = "spectral"
#cluster_model = "dbscan"

img = Image.open("images/" + "4155610_198_masked" + ".png")
img = np.array(img, dtype=np.uint8)

img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

sh = np.shape(img)
width = np.shape(img)[0]
length = np.shape(img)[1]

img_seg = []
img_seg_coords = []
for i in range(0,width):
    for j in range(0,length):
        pixel = img[i][j]
        #if pixel[3] != 0:
        if not(pixel[0] == 0 and pixel[1] == 128 and pixel[2] == 128):
            img_seg.append(pixel)
            #img_seg_coords.append((i,j))
            img_seg_coords.append(True)
        else:
            img_seg_coords.append(False)

img_pixels = np.array(img_seg)
img_seg_coords = np.array(img_seg_coords)
#img_seg_coords = img_seg_coords.reshape((width,length))

# create cluster model
model = KMeans(n_clusters=4)
model.fit(img_pixels)
preds = model.predict(img_pixels)

if cluster_model == "optics":
    model = OPTICS(eps=0.8, min_samples=10)
    model.fit(img_pixels)
    preds = model.fit_predict(img_pixels)

if cluster_model == "spectral":
    model = SpectralClustering(n_clusters=3)
    preds = model.fit_predict(img_pixels)

if cluster_model == "dbscan":
    model = DBSCAN(eps=0.60, min_samples=3)
    preds = model.fit_predict(img_pixels)
    
if cluster_model == "agglomerative":
    model = AgglomerativeClustering(n_clusters=3)
    preds = model.fit_predict(img_pixels)
    
# get cluster centroids
clf = NearestCentroid()
clf.fit(img_pixels, preds)

# assign centroid of each pixel based on cluster
for p in unique(preds):
    img_pixels[preds == p] = clf.centroids_[p]

# messy code to reassign pixel values, unsure how to do this properly with numpy 
img = img.reshape(-1, img.shape[-1])
i = 0
j = 0
for c in img_seg_coords:
    if(c == True):
        img[i] = img_pixels[j]
        j+=1
    i+=1
img = img.reshape(sh)

img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

cv2.imwrite("test.png",img)



cv2.imshow('Mask', test)
cv2.waitKey(0)
cv2.imwrite("test.png",test)
test2 = test.reshape(-1, test.shape[-1])
test3 = img.reshape(-1, img.shape[-1])
    
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = KMeans(n_clusters=2)
model = OPTICS(eps=0.8, min_samples=10)
# fit the model
model.fit(vals)
# assign a cluster to each example
yhat = model.fit_predict(vals)
yhat = model.predict(vals)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()