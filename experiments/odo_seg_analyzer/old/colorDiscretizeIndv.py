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

def colorDiscretizeIndv(rgba_img,img_name,cluster_model="kmeans",out_dir=None,show=False):
    img = rgba_img

    if(show):
        cv2.imshow('Input Image', img)
        cv2.waitKey(0)

    # convert to cielab for more "perceptual" clustering
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # save initial image shape
    sh = np.shape(img)
    width = np.shape(img)[0]
    length = np.shape(img)[1]

    # get pixels in segment (vs in the black/transparent background)
    # save the coordinates of those pixels as well
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

    # create cluster model
    if cluster_model == "kmeans":
        model = KMeans(n_clusters=4)
        model.fit(img_pixels)
        preds = model.predict(img_pixels)

    if cluster_model == "optics":
        model = OPTICS(eps=0.8, min_samples=10)
        model.fit(img_pixels)
        preds = model.fit_predict(img_pixels)

    if cluster_model == "spectral":
        model = SpectralClustering(n_clusters=3)
        model.fit(img_pixels)
        preds = model.fit_predict(img_pixels)

    if cluster_model == "dbscan":
        model = DBSCAN(eps=0.60, min_samples=3)
        model.fit(img_pixels)
        preds = model.fit_predict(img_pixels)
        
    if cluster_model == "agglomerative":
        model = AgglomerativeClustering(n_clusters=3)
        model.fit(img_pixels)
        preds = model.fit_predict(img_pixels)
        
    # get cluster centroids
    clf = NearestCentroid()
    clf.fit(img_pixels, preds)

    # assign centroid of each pixel
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

    # convert back RGBA for write 
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    if(show):
        cv2.imshow('Discretized Image', img)
        cv2.waitKey(0)

    cv2.imwrite(out_dir + "/" + img_name + "_discrete.png",img)