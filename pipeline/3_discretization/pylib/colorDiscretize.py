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
from helpers import showImages

# colorDiscretize takes a list of tuples of rgba imgs, masks, and names
def colorDiscretize(rgba_imgs_masks_ids, group_cluster=False, by_contours=True, erode_contours=True, erode_kernel_size=2, dilate_multiplier = 2, min_contour_area=10,
                    cluster_model="kmeans", nclusters=3, cluster_min_samples=7, verticalize=True, resize=True, show=False, export_dir = "../../data/patterns/individuals"):
    # if clustering by group, gather cluster values for all individuals first
    # highly redundant code, consider making a function that outputs contours and contour pixel means for an image
    if(group_cluster):
        group_cluster_values = np.empty((0,4), float)

        # redundant loop more or less repeated below
        for img, mask, id in rgba_imgs_masks_ids:
            # convert img to grey
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # get threshold image
            thresh_img = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
            if erode_contours:
                kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
                thresh_img = cv2.erode(thresh_img, kernel)
                thresh_img = cv2.dilate(thresh_img, kernel * dilate_multiplier)

            # find contours
            contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            # keep contours above the contour area threshold
            new_conts = list()
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_contour_area:
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
                #img_lab = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                #img_lab = cv2.cvtColor(img_lab, cv2.COLOR_RGB2LAB)

                # add the mean values of these pixels to contour_means
                contour_means.append(np.mean(img[pts[0], pts[1]], axis=0)) #img_lab

            # get the contour of the whole segment
            mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # create a mask image that contains the contour filled in
            cimg = np.zeros_like(img)
            cv2.drawContours(cimg, mask_contours, 0, color=255, thickness=-1)

            # for every NON-PRIMARY MASK contour
            for i in range(len(contours)):
                # create a mask image that contains the contour filled in
                cv2.drawContours(cimg, contours, i, color=[0, 0, 0, 0], thickness=-1)

            # Access the image pixels and create a 1D numpy array then add to list
            primary_contour_points = np.where(cimg == 255)

            # add primary contour to contour means
            contour_means.append(np.mean(img[primary_contour_points[0], primary_contour_points[1]], axis=0))
            # add new contour means to all_cluster_values holding all cluster values for the group
            group_cluster_values = np.vstack((group_cluster_values,np.array(contour_means)))

    # for every segment...
    for img, mask, id in rgba_imgs_masks_ids:
        start_img = np.copy(img)

        # if clustering by contours...
        if(by_contours):
            # convert img to grey
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # get threshold image
            # ret, thresh_img = cv2.threshold(img_grey, contour_threshold, 255, cv2.THRESH_BINARY)
            thresh_img = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
            start_thresh = np.copy(thresh_img)

            if erode_contours:
                #resized_thresh = cv2.resize(thresh_img, (thresh_img.shape[1] * 2,thresh_img.shape[0] * 2))
                kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
                #resized_thresh = cv2.erode(resized_thresh,kernel)
                thresh_img = cv2.erode(thresh_img,kernel)
                thresh_img = cv2.dilate(thresh_img, kernel * dilate_multiplier)
                #thresh_img = cv2.resize(resized_thresh,(thresh_img.shape[1],thresh_img.shape[0]))
                showImages(show,[start_thresh,thresh_img],["Original", "Eroded"])

            # find contours
            contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            # keep contours above the contour area threshold
            new_conts = list()
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_contour_area:
                    new_conts.append(cnt)
            hull_contours = new_conts
            contours = tuple(new_conts)

            # save contours to show later
            contour_img = np.zeros_like(img)
            # possible strategy - take convex hulls of contours
            #for i in range(0,len(hull_contours)):
            #    hull_contours[i] = cv2.convexHull(hull_contours[i])
            #hull_contours = tuple(hull_contours)
            cv2.drawContours(contour_img,contours,-1,  color=255,  thickness=-1) #hull_contours

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
                #img_lab = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                #img_lab = cv2.cvtColor(img_lab, cv2.COLOR_RGB2LAB)

                # add the mean values of these pixels to contour_means
                contour_means.append(np.mean(img[pts[0], pts[1]], axis=0)) #img_lab

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
            model = KMeans(n_clusters=nclusters)
            if group_cluster:
                model.fit(group_cluster_values)
            else:
                model.fit(cluster_values)
            preds = model.predict(cluster_values)

        if cluster_model == "optics":
            model = OPTICS(min_samples=cluster_min_samples)
            if group_cluster:
                model.fit(group_cluster_values)
            else:
                model.fit(cluster_values)
            preds = model.fit_predict(cluster_values)

        if cluster_model == "spectral":
            model = SpectralClustering(n_clusters=nclusters)
            if group_cluster:
                model.fit(group_cluster_values)
            else:
                model.fit(cluster_values)
            preds = model.fit_predict(cluster_values)

        if cluster_model == "dbscan":
            model = DBSCAN(eps=0.50, min_samples=cluster_min_samples)
            if group_cluster:
                model.fit(group_cluster_values)
            else:
                model.fit(cluster_values)
            preds = model.fit_predict(cluster_values)

        if cluster_model == "agglomerative":
            model = AgglomerativeClustering(n_clusters=nclusters)
            if group_cluster:
                model.fit(group_cluster_values)
            else:
                model.fit(cluster_values)
            preds = model.fit_predict(cluster_values)

        # TODO change all this to get cluster centroid values from fitted model,then assign to each cluster value the centroid value, then draw contours back on using those values
        # get cluster centroids
        clf = NearestCentroid()
        print(unique(preds))
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

            # doesn't work just yet
            if resize:
                height = img.shape[0]
                width = img.shape[1]
                target_area = 50000 # equivalent to a typical 100x500 segment
                hw_multiplier = int(target_area / (height + width))
                img = cv2.resize(img,[height * hw_multiplier,width * hw_multiplier])

        showImages(show,[start_img,start_thresh,contour_img,img],["Image Segment","Thresholded Image", "Segment Contours","Discretized Segment"])
        if verticalize:
            img = rotateToVertical([img],show=True)
        cv2.imwrite(export_dir + "/" + id + "_pattern.jpg", img[0])