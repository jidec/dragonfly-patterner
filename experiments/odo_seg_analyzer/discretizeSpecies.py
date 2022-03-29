from numpy import unique
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import numpy as np
import cv2
from sklearn.neighbors import NearestCentroid
import os

def discretizeSpecies(species_name, img_dir, by_contours=True, cluster_model="kmeans", out_dir=None, show=False):
    # get all image names of the species
    species_image_names = []
    all_contours = []
    all_contour_means = []

    for img_name in species_image_names:
        img = cv2.imread(os.path.join(img_dir, img_name)) #_x.png
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

        # add new contours to all contours
        all_contours = all_contours + contours

        contours = tuple(new_conts)
        # initialize empty list to hold the mean pixel values of each contour
        contour_means = []

        # get the contour of the whole segment
        thresh = 3
        # get mask using threshold
        ret, mask = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
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
        # add this set of contour means to all contour mean
        all_contour_means.append(contour_means)

    cluster_values = all_contours
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

    # go back to every image, and draw its contours back on