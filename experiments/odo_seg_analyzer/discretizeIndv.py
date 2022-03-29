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

def discretizeIndv(rgba_img_and_mask, img_name, by_contours=True, cluster_model="kmeans", out_dir=None, show=False):
    img = rgba_img_and_mask[0]
    mask = rgba_img_and_mask[1]

    if (show):
        cv2.imshow('Input Image', img)
        cv2.waitKey(0)

    # if clustering by contours...
    if(by_contours):
        # convert img to grey
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # set a thresh
        thresh = 100
        # get threshold image
        ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
        # find contours
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
            contour_means.append(np.mean(img[pts[0], pts[1]], axis=0))

        # get the contour of the whole segment
        mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # create a mask image that contains the contour filled in
        cimg = np.zeros_like(img)
        cv2.drawContours(cimg, mask_contours, 0, color=255, thickness=-1)

        # for every NON-PRIMARY MASK contour
        for i in range(len(contours)):
            # Create a mask image that contains the contour filled in
            cv2.drawContours(cimg, contours, i, color=[0, 0, 0, 0], thickness=-1)

        # Access the image pixels and create a 1D numpy array then add to list
        primary_contour_points = np.where(cimg == 255)

        # add primary contour to contour means
        contour_means.append(np.mean(img[primary_contour_points[0], primary_contour_points[1]], axis=0))
        cluster_values = np.array(contour_means)
    # if clustering by pixels...
    else:
        # convert to cielab for more "perceptual" clustering
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
        for i in range(0, width):
            for j in range(0, length):
                pixel = img[i][j]
                # if pixel[3] != 0:
                if not (pixel[0] == 0 and pixel[1] == 128 and pixel[2] == 128):
                    img_seg.append(pixel)
                    # img_seg_coords.append((i,j))
                    img_seg_coords.append(True)
                else:
                    img_seg_coords.append(False)
        cluster_values = np.array(img_seg)
        cluster_values_coords = np.array(img_seg_coords)

    # create cluster model
    if cluster_model == "kmeans":
        model = KMeans(n_clusters=4)
        model.fit(cluster_values)
        preds = model.predict(cluster_values)

    if cluster_model == "optics":
        model = OPTICS(eps=0.8, min_samples=10)
        model.fit(cluster_values)
        preds = model.fit_predict(cluster_values)

    if cluster_model == "spectral":
        model = SpectralClustering(n_clusters=3)
        model.fit(cluster_values)
        preds = model.fit_predict(cluster_values)

    if cluster_model == "dbscan":
        model = DBSCAN(eps=0.60, min_samples=3)
        model.fit(cluster_values)
        preds = model.fit_predict(cluster_values)

    if cluster_model == "agglomerative":
        model = AgglomerativeClustering(n_clusters=3)
        model.fit(cluster_values)
        preds = model.fit_predict(cluster_values)

    # get cluster centroids
    clf = NearestCentroid()
    clf.fit(cluster_values, preds)

    # assign centroid of each contour or pixel
    for p in unique(preds):
        cluster_values[preds == p] = clf.centroids_[p]

    # if clustering by contours...
    if(by_contours):
        # draw contours with mean colors
        img = np.zeros_like(img)

        for i in range(len(contours)):
            # Create a mask image that contains the contour filled in
            img = cv2.drawContours(img, contours, i, color=contour_means[i], thickness=-1)

        # reassign primary contour cluster centroid to image
        img[primary_contour_points[0], primary_contour_points[1]] = contour_means[len(contour_means) - 1]
    # if clustering by pixels...
    else:
        # messy code to reassign pixel values, unsure how to do this properly with numpy
        img = img.reshape(-1, img.shape[-1])
        i = 0
        j = 0
        for c in img_seg_coords:
            if (c == True):
                img[i] = cluster_values[j]
                j += 1
            i += 1
        img = img.reshape(sh)

        # convert back RGBA for write
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    if (show):
        cv2.imshow('Discretized Image', img)
        cv2.waitKey(0)

    cv2.imwrite(out_dir + "/" + img_name + "_discrete.png", img)