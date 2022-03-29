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
from colorDiscretizeIndv import colorDiscretizeIndv

def colorDiscretizeIndvContours(rgba_img_and_mask,img_name,cluster_model="kmeans",out_dir=None,show=False):
    
    # get img and mask from input
    img = rgba_img_and_mask[0]
    mask = rgba_img_and_mask[1]

    if (show):
        cv2.imshow('Input Image', img)
        cv2.waitKey(0)

    #convert img to grey
    img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #set a thresh
    thresh = 100
    #get threshold image
    ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    #find contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    threshold_area = 10
    # keep contours above the contour area threshold
    new_conts = list()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > threshold_area:
            new_conts.append(cnt)
    contours = tuple(new_conts)
    
    # initialize empty list to hold the mean pixel values of each contour
    contour_means = []
    
    # for each list of contour points...
    for i in range(len(contours)):
        # create a mask image that contains the contour filled in
        cimg = np.zeros_like(img)
        cv2.drawContours(cimg, contours, i, color=255, thickness=-1)
    
        # access the image pixels and create a 1D numpy array
        pts = np.where(cimg == 255)
        # add the mean values of these pixels to contour_means
        contour_means.append(np.mean(img[pts[0], pts[1]],axis=0))
    
    # get the contour of the whole segment
    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # create a mask image that contains the contour filled in
    cimg = np.zeros_like(img)
    cv2.drawContours(cimg, mask_contours, 0, color=255, thickness=-1)
    
    # for every NON-PRIMARY MASK contour 
    for i in range(len(contours)):
        # Create a mask image that contains the contour filled in
        cv2.drawContours(cimg, contours, i, color=[0,0,0,0], thickness=-1)
    
    # Access the image pixels and create a 1D numpy array then add to list
    primary_contour_points = np.where(cimg == 255)
    
    # add primary contour to contour means
    contour_means.append(np.mean(img[primary_contour_points[0], primary_contour_points[1]],axis=0))
    contour_means = np.array(contour_means)
    
    # create cluster model
    model = KMeans(n_clusters=3)
    model.fit(contour_means)
    preds = model.predict(contour_means)
    
    # get cluster centroids
    clf = NearestCentroid()
    clf.fit(contour_means, preds)
    
    # assign centroid of each pixel
    for p in unique(preds):
        contour_means[preds == p] = clf.centroids_[p]
    
    # draw contours with mean colors 
    cimg = np.zeros_like(img)
    
    for i in range(len(contours)):
        # Create a mask image that contains the contour filled in
        cimg = cv2.drawContours(cimg, contours, i, color=contour_means[i], thickness=-1)
    
    # reassign primary contour cluster centroid to image 
    cimg[primary_contour_points[0], primary_contour_points[1]] = contour_means[len(contour_means) - 1]
    
    cv2.imshow('test', cimg)
    cv2.waitKey(0)
    
    cv2.imwrite(img_name + "_discrete.png",cimg)
