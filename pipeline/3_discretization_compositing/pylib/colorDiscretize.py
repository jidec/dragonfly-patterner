import pandas as pd
from numpy import unique
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import numpy as np
import cv2
from sklearn.neighbors import NearestCentroid
from rotateToVertical import rotateToVertical
from showImages import showImages

# problems before I forget
# image ids getting modified in methods, create copy at beginning of each method - done I think
# bad or empty images slipping through
def colorDiscretize(image_ids, group_cluster_records_col = None, group_cluster_raw_ids = False, blur_kernel_size=0,
                    by_contours=True, erode_contours_kernel_size=0, dilate_multiplier = 2, min_contour_pixel_area=10,
                    cluster_model="kmeans", nclusters=3, cluster_min_samples=7,
                    print_steps=True, print_details=False, show=False, proj_dir="../.."):
    """
        Discretize (AKA quantize or recolorize) continuously shaded organismal segments to discrete patterns

        :param List image_ids: the imageIDs (image names) to extract segments from
        :param bool group_cluster_records_col: the column in records to group images before clustering i.e "species"
            can also be "speciesMorph" which gets merged from inferences
        :param bool by_contours: whether to cluster by contours instead of by pixels
        :param int erode_contours_kernel_size: the size of the kernel with which to erode contours, no erosion if 0
        :param float dilate_multiplier: not sure what this does LMAO
        :param int min_contour_pixel_area: the minimum number of pixels in a contour to include it as a pattern element in clustering
        :param str cluster_model: the type of clustering to perform, either "kmeans" or "optics" for now
        :param int nclusters: the number of clusters (if using an algo that requires specifying)
        :param int cluster_min_samples: a parameter of OPTICS clustering

        :param bool show: whether or not to show image processing outputs and intermediates
        :param bool print_steps: whether or not to print processing step info after they are performed
        :param bool proj_dir: the path to the project directory containing /data and /trainset_tasks folders
    """
    image_ids = image_ids.copy()

    # gather either contour data or pixel data for every image
    contour_or_pixel_data = []
    if print_steps: print("Started gathering pixel or contour data for each image")
    for id in image_ids:
        img = cv2.imread(proj_dir + "/data/segments/" + id + "_segment.png",cv2.IMREAD_UNCHANGED)

        # blur if specified
        if blur_kernel_size != 0:
            img = cv2.blur(img,blur_kernel_size)
            if print_details: print("Blurred input image")

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if print_details: print("Loaded segment for ID " + id)
        if img is None:
            continue
        if by_contours:
            contour_or_pixel_data.append(getContoursAndContourPixelMeans(img,erode_contours_kernel_size,dilate_multiplier,min_contour_pixel_area,show))
            if print_details: print("Gathered contour data")
        else:
            contour_or_pixel_data.append(getPixelsAndPixelCoords(img))
            if print_details: print("Gathered pixel data")

    # cluster by group
    if group_cluster_records_col is not None:
        if print_steps: print("Started clustering by group")
        # load records and keep those with matching ids
        records = pd.read_csv("data/inatdragonflyusa_records.csv")
        records = records[records["imageID"].isin(image_ids),:]

        # loop through unique groups and update contour data using clustered centroids
        unique_groups = unique(records[group_cluster_records_col])
        for g in unique_groups:
            # group_ids = records.query(group_cluster_records_col + '==' + g)["imageID"]

            # get indices for the group
            group_indices = records.index[records[group_cluster_records_col]==g].tolist()

            # get data for the group
            group_data = contour_or_pixel_data[group_indices]

            # get 2nd element of each tuple, the pixel means
            group_contour_means = [c[1] for c in group_data]

            # cluster and get new values
            clustered_values = getClusterCentroids(group_contour_means,cluster_model,nclusters,cluster_min_samples)

            # replaced old values in group data with new values
            for index, d in enumerate(group_data):
                group_data[index] = tuple(d[0] + clustered_values[index] + d[2])

            # add group data back to main data
            contour_or_pixel_data[group_indices] = group_data

    elif group_cluster_raw_ids:
        print("gc raw")
        # get all
        print(contour_or_pixel_data[1][1])
        group_means = [cpd[1] for cpd in contour_or_pixel_data]
        group_means = sum(group_means, [])
        print("Len group means " + str(len(group_means)))
        clustered_values = getClusterCentroids(group_means, cluster_model, nclusters, cluster_min_samples)
        print("Len clust vals " + str(len(clustered_values)))
        i = 0
        for index, cpd in enumerate(contour_or_pixel_data):
            cpd = list(cpd)
            print("Len cpd " + str(len(cpd[1])))
            # set cpd mean values to clustered values
            n_values = len(cpd[1])
            cpd[1] = clustered_values[i:(n_values + i)]
            i = i + n_values
            cpd = tuple(cpd)
            print("Len cpd 2 " + str(len(cpd[1])))
            contour_or_pixel_data[index] = cpd

    # if not grouping, just cluster at the level of individual images
    else:
        for index, cpd in enumerate(contour_or_pixel_data):
            cpd = list(cpd)
            clustered_values = getClusterCentroids(cpd[1], cluster_model, nclusters, cluster_min_samples)
            cpd[1] = clustered_values
            cpd = tuple(cpd)
            contour_or_pixel_data[index] = cpd

    # loop over cluster-centered data
    i = 0
    for cpd in contour_or_pixel_data:
        img = np.zeros(shape=cpd[2])
        if by_contours:
            contours = cpd[0]
            contour_means = cpd[1]
            # draw contours with mean colors
            for j in range(len(contours)):
                # Create a mask image that contains the contour filled in
                img = cv2.drawContours(img, contours, j, color=contour_means[j], thickness=-1)
        else:
            # messy code to reassign pixel values, unsure how to do this properly with numpy
            pixel_coords = cpd[0]
            pixel_vals = cpd[1]

            for index, p in enumerate(pixel_vals):
                img[pixel_coords[index]] = p
        img = img.reshape(cpd[2])

        cv2.imwrite(proj_dir + "/data/patterns/" + image_ids[i] + "_pattern.png", img)
        if(print_details): print("Wrote to " + proj_dir + "/data/patterns/" + image_ids[i] + "_pattern.png")

        print(i)
        i = i + 1

# key function that is given an image (png RGBA segment) and returns a tuple three lists
#   the first list containing contours (lists of contour points),
#   the second list containing pixel means for each contour
#   the third list containing image dimensions for reconstruction
def getContoursAndContourPixelMeans(img,erode_contours_kernel_size,dilate_multiplier,min_contour_pixel_area,show):
    # convert img to grey
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get threshold image to use when finding contours
    thresh_img = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    # erode
    if erode_contours_kernel_size != 0:
        kernel = np.ones((erode_contours_kernel_size, erode_contours_kernel_size), np.uint8)
        thresh_img = cv2.erode(thresh_img, kernel)
        thresh_img = cv2.dilate(thresh_img, kernel * dilate_multiplier)

    # find contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # keep contours above the contour area threshold
    new_conts = list()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_contour_pixel_area:
            new_conts.append(cnt)
    # append these new contours to all contours
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

        # convert image to CIELAB
        # img_lab = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        # img_lab = cv2.cvtColor(img_lab, cv2.COLOR_RGB2LAB)

        # add the mean values of these pixels to contour_means
        contour_means.append(np.mean(img[pts[0], pts[1]], axis=0))  # img_lab

    # start contour image as containing the whole segment
    ret,thresh_img_mask = cv2.threshold(img_grey,1,255,cv2.THRESH_BINARY)
    #thresh_img_mask = cv2.cvtColor(thresh_img_mask, cv2.COLOR_BGR2GRAY)

    # get the contour of the whole segment
    primary_contours, hierarchy = cv2.findContours(thresh_img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # create a mask image that contains the contour filled in
    cont_img = np.zeros_like(img)
    for i in range(len(primary_contours)):
        cv2.drawContours(cont_img, primary_contours, i, color=[255,255,255,255], thickness=-1)

    showImages(show, [cont_img], ["Primary contour"])

    # for every NON-PRIMARY MASK contour
    # fill in contour image
    for i in range(len(contours)):
        # create a mask image that contains the contour filled in
        cv2.drawContours(cont_img, contours, i, color=[0, 0, 0, 0], thickness=-1)

    showImages(show, [cont_img], ["Primary contour backfilled with other contours"])

    # Access the image pixels and create a 1D numpy array then add to list
    primary_contour_points = np.where(cont_img != 0)
    primary_cont_mean = np.mean(img[primary_contour_points[0], primary_contour_points[1]], axis=0)

    contours = list(contours)
    # add primary contour mean to contour means
    for i in range(len(primary_contours)):
        contours.insert(0,primary_contours[i])
        contour_means.insert(0,primary_cont_mean)

    return (contours,contour_means,img.shape)

# return a tuple containing pixel coords, pixel values, and the shape of the image
def getPixelsAndPixelCoords(img):
    pixels = []
    pixel_coords = []
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            pixel = img[i][j]
            if not (pixel[3] == 0):
                pixels.append(pixel)
                pixel_coords.append((i, j))
    return (pixel_coords,pixels,img.shape)

# given a set of values (either pixel values or contour means), assign cluster centroid values to them and return
def getClusterCentroids(values,cluster_model,nclusters,cluster_min_samples):

    # create cluster model
    if cluster_model == "kmeans":
        model = KMeans(n_clusters=nclusters)
        model.fit(values)
        preds = model.predict(values)

    if cluster_model == "optics":
        model = OPTICS(min_samples=cluster_min_samples)
        model.fit(values)
        preds = model.fit_predict(values)

    if cluster_model == "spectral":
        model = SpectralClustering(n_clusters=nclusters)
        model.fit(values)
        preds = model.fit_predict(values)

    if cluster_model == "dbscan":
        model = DBSCAN(eps=0.50, min_samples=cluster_min_samples)
        model.fit(values)
        preds = model.fit_predict(values)

    if cluster_model == "agglomerative":
        model = AgglomerativeClustering(n_clusters=nclusters)
        model.fit(values)
        preds = model.fit_predict(values)

    # get cluster centroids
    clf = NearestCentroid()
    clf.fit(values, preds)
    values = np.array(values)

    # assign centroid of each contour or pixel
    for p in unique(preds):
        values[preds == p] = clf.centroids_[p]

    return values