import pandas as pd
from numpy import unique
from sklearn.cluster import KMeans, OPTICS, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np
import cv2
from sklearn.neighbors import NearestCentroid
from rotateToVertical import rotateToVertical
from showImages import showImages
from plotPixels import plotPixels
from matplotlib import pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from kneed import DataGenerator, KneeLocator
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import colorsys
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# problems before I forget
# image ids getting modified in methods, create copy at beginning of each method - done I think
# bad or empty images slipping through
def colorDiscretize(image_ids, preclustered = False, group_cluster_records_col = None, group_cluster_raw_ids = False,
                    by_contours=False, erode_contours_kernel_size=0, dilate_multiplier = 2, min_contour_pixel_area=10,
                    cluster_model="gaussian_mixture", nclusters=None, nclust_metric = "ch", cluster_eps=0.5, cluster_min_samples=4,
                    colorspace = None, scale = False, use_positions=False, downweight_axis=None, upweight_axis=None,
                    bilat_blur = True, vert_resize = 400,
                    print_steps=True, print_details=False, write_subfolder= "", show=False, proj_dir="../.."):
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
    populated_ids = []
    #print(image_ids)
    # gather either contour data or pixel data for every image
    contour_or_pixel_data = []
    if (print_steps):
        if preclustered:
            print("Estimated time " + str(len(image_ids)/30) + " minutes at default cluster mode, metric, and vert resize")
        else:
            print("Estimated time " + str((len(image_ids)/30)/3) + " minutes at default cluster mode, metric, and vert resize")

    if print_steps: print("Started gathering pixel or contour data for each image")
    for id in image_ids:
        if not preclustered:
            img = cv2.imread(proj_dir + "/data/segments/" + id + "_segment.png",cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imread(proj_dir + "/data/patterns/" + id + "_pattern.png", cv2.IMREAD_UNCHANGED)

        showImages(show,[img],"Segment")

        # blur if specified
        #if blur_kernel_size != 0:
        #    img = cv2.blur(img,blur_kernel_size)
        #    if print_details: print("Blurred input image")

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if print_details: print("Loaded segment for ID " + id)
        if img is None:
            if print_details: print("Image is empty - skipping")
            #image_ids.remove(id)
            continue
        if bilat_blur:
            alpha = img[:, :, 3]
            img = img[:, :, :3]
            img = cv2.bilateralFilter(img, d=15, sigmaColor=100, sigmaSpace=60)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            img[:, :, 3] = alpha
            showImages(show,[img],['Blurred Image'])

        if colorspace == "hls":
            alpha = img[:, :, 3] # save alpha
            img = img[:, :, :3] # remove alpha
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
            img = np.dstack((img, alpha))
        elif colorspace == "lab":
            alpha = img[:, :, 3]  # save alpha
            img = img[:, :, :3]  # remove alpha
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            img = np.dstack((img, alpha)) # return alpha (necessary for excluding pixels)

        if vert_resize is not None:
            dim_mult = vert_resize / img.shape[0]
            img = cv2.resize(img, dsize=(int(img.shape[1] * dim_mult),vert_resize))
        if show:
            plotPixels(img)

        if by_contours:
            contour_or_pixel_data.append(getContoursAndContourPixelMeans(img,erode_contours_kernel_size,dilate_multiplier,min_contour_pixel_area,show))
            if print_details: print("Gathered contour data")
        else:
            contour_or_pixel_data.append(getPixelsAndPixelCoords(img))
            if print_details: print("Gathered pixel data")
        populated_ids.append(id)


    #print(len(contour_or_pixel_data))

    # cluster by group
    if group_cluster_records_col is not None:
        if print_steps: print("Started group clustering by a records column (probably species or clade)")
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
            clustered_values = getClusterCentroids(group_contour_means,cluster_model,nclusters,cluster_eps,cluster_min_samples,scale,use_positions,downweight_axis,upweight_axis,preclustered,nclust_metric,img,show)

            # replaced old values in group data with new values
            for index, d in enumerate(group_data):
                group_data[index] = tuple(d[0] + clustered_values[index] + d[2])

            # add group data back to main data
            contour_or_pixel_data[group_indices] = group_data

    elif group_cluster_raw_ids:
        if print_steps: print("Group clustering all IDs together")
        #print("gc raw")
        # get all
        #print(contour_or_pixel_data[1][1])
        group_means = [cpd[1] for cpd in contour_or_pixel_data]
        group_coords = [cpd[0] for cpd in contour_or_pixel_data]
        group_means = sum(group_means, [])
        print("Len group means " + str(len(group_means)))
        clustered_values = getClusterCentroids(group_means, group_coords, cluster_model, nclusters, cluster_eps, cluster_min_samples,scale,use_positions,downweight_axis,upweight_axis,preclustered,nclust_metric,img,show)
        print("Len clust vals " + str(len(clustered_values)))
        i = 0
        for index, cpd in enumerate(contour_or_pixel_data):
            cpd = list(cpd)
            #print("Len cpd " + str(len(cpd[1])))
            # set cpd mean values to clustered values
            n_values = len(cpd[1])
            cpd[1] = clustered_values[i:(n_values + i)]
            i = i + n_values
            cpd = tuple(cpd)
            #print("Len cpd 2 " + str(len(cpd[1])))
            contour_or_pixel_data[index] = cpd

    # if not grouping, just cluster at the level of individual images
    else:
        if print_steps: print("Clustering individually")
        for index, cpd in enumerate(contour_or_pixel_data):
            cpd = list(cpd)
            if print_details: print("Clustering next image")
            clustered_values = getClusterCentroids(cpd[1], cpd[0], cluster_model, nclusters, cluster_eps, cluster_min_samples,scale,use_positions,downweight_axis,upweight_axis,preclustered,nclust_metric,img,show)
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
                #p = np.array(colorsys.hls_to_rgb(p[0]/255,p[1]/255,p[2]/255))
                p = np.array((p[0], p[1], p[2]))
                p = np.append(p,255) # add opaque alpha
                #print(p)
                img[pixel_coords[index]] = p

        img = img / 255

        if colorspace == "hls":
            # convert back to RGB from HLS
            alpha = img[:, :, 3]  # save alpha
            img = img[:, :, :3]  # remove alpha
            img = img.astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB_FULL)
            img = np.dstack((img, alpha))
        elif colorspace == "lab":
            alpha = img[:, :, 3]
            img = img[:, :, :3]
            img = img.astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)
            img = np.dstack((img, alpha))

        #alpha = img[:, :, 3]
        #img = img[:, :, :3]
        #img = img.astype(np.float32)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        #img[:, :, 3] = alpha

        img = img * 255

        # reshape to original dims
        #img = img.reshape(cpd[2])

        if show:
            cv2.imshow("i", img)
            cv2.waitKey(0)

        #showImages(show,[img],['Discretized'])
        if write_subfolder != "":
            if not os.path.exists(proj_dir + "/data/patterns/" + write_subfolder):
                os.mkdir(proj_dir + "/data/patterns/" + write_subfolder)
            write_subfolder = write_subfolder + "/"
        write_target = proj_dir + "/data/patterns/" + write_subfolder + populated_ids[i] + "_pattern.png"
        cv2.imwrite(write_target, img)
        if(print_details): print("Wrote to " + write_target)

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

        # convert image
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        #if color_fun is not None: #cv2.COLOR_RGB2LAB, RGB2HSV
        #    img = cv2.cvtColor(img, color_fun)



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
                p = np.array([pixel[0],pixel[1],pixel[2]])
                #print(p)
                #if color_fun == cv2.COLOR_RGB2HLS:
                    #p[0] = p[0] / 180
                    #p[1] = p[1] / 255
                    #p[2] = p[2] / 255
                #else:
                    #p = p / 255
                #p = np.array(pixel[0])

                pixels.append(p) # append 0-1
                pixel_coords.append((i, j))
    return (pixel_coords,pixels,img.shape)

# given a set of values (either pixel values or contour means), assign cluster centroid values to them and return
def getClusterCentroids(values,coords,cluster_model,nclusters,cluster_eps,cluster_min_samples,scale,use_positions,downweight_axis,upweight_axis,preclustered,nclust_metric,img,show):

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
        model = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples)
        model.fit(values)
        preds = model.fit_predict(values)

    if cluster_model == "agglomerative":
        model = AgglomerativeClustering(n_clusters=nclusters)
        model.fit(values)
        preds = model.fit_predict(values)

    if cluster_model == "gaussian_mixture":

        start_values = values
        # convert to np array
        values = np.array(values)
        #print(X)
        #print(X.shape)
        # using range of variation within an image, rather than of all possible ranges including unrealistic ones
        if use_positions:
            xpos = [p[0] for p in coords]
            ypos = [p[1] for p in coords]
            #print(np.shape(X)[1])
            values = np.insert(values, np.shape(values)[1], xpos, axis=1)
            values = np.insert(values, np.shape(values)[1], ypos, axis=1)
        if scale:
            #scaler = StandardScaler(with_mean=False).fit(X)
            scaler = MinMaxScaler().fit(values)
            #scaler0 = StandardScaler().fit(X[:, 0])
            #print(X[:, 0])
            #scaler1 = StandardScaler().fit(X[:, 1])
            #scaler2 = StandardScaler().fit(X[:, 2])
            values = scaler.transform(values)

            #X = X / X.max(axis=0)
        if downweight_axis is not None:
            values[:, downweight_axis] = values[:, downweight_axis] / 2
        if upweight_axis is not None:
            values[:,upweight_axis] = values[:, upweight_axis] * 2

        if nclusters is None:
            # create a vector of ns to try for knee assessment
            n_components = np.arange(3, 6)
            #print(n_components)
            # create models for each n and get aics

            if preclustered:
                X = np.unique(values, axis=0)
                print(X)
            else:
                X = values
            models = [GaussianMixture(n).fit(X) for n in n_components]
            #aics = [m.aic(X) for m in models]

            preds = [m.fit_predict(X) for m in models]

            if nclust_metric == "ch":
                scores = [calinski_harabasz_score(X,p) for p in preds]
                nclusters = n_components[np.argmax(scores)]
            elif nclust_metric == "db":
                scores = [davies_bouldin_score(X, p) for p in preds]
                nclusters = n_components[np.argmin(scores)]
            #print(ch_scores)
            #plt.plot(n_components, ch_scores, label='CH score')
            #plt.show()
            #nclusters = n_components[np.argmax(scores)]
        #model = models[np.argmax(ch_scores)]
        #print(model)
        #plt.legend(loc='best')
        #plt.xlabel('n_components');
        # locate knee
        #kneedle = KneeLocator(n_components,aics, S=2, curve="convex", direction="decreasing")
        #if show:
        #    kneedle.plot_knee()
        #    plt.show()
        #nclusters = round(kneedle.knee, 1)
        #print(nclusters)

        #plt.plot(n_components, aics, label='AIC')
        #plt.legend(loc='best')
        #plt.xlabel('n_components');
        #plt.show()
        #print(values)
        #values = np.delete(values, 3, 1)

        model = GaussianMixture(n_components=nclusters)

        # if preclustered, fit clusters only unique pixel colors (i.e. each pre-existing color cluster)
        if preclustered:
            model.fit(np.unique(values, axis=0))
        # otherwise fit clusters using all values
        else:
            model.fit(values)
        # regardless, predict using values
        preds = model.fit_predict(values)

        #model = KMeans()
        #visualizer = KElbowVisualizer(
        #    model, k=(2, 6), metric='calinski_harabasz', timings=False
        #)
        #visualizer.fit(X)  # Fit the data to the visualizer
        #visualizer.show()  # Finalize and render the figure


    # get cluster centroids
    clf = NearestCentroid()
    clf.fit(values, preds)
    values = np.array(values)

    centroids = clf.centroids_
    if scale:
        centroids = scaler.inverse_transform(centroids)
        #print(centroids)

    if show:
        pix_centroids = centroids[:, :3]
        if preclustered:
            fig = plt.figure()
            ax = Axes3D(fig)
            uniq = np.unique(values,axis=0)
            uniq = uniq[:, [2, 1, 0]]
            ax.scatter(uniq[:,0], uniq[:,1],uniq[:,2], c=uniq/255,alpha=0.5)
            #ax.scatter(pix_centroids[:, 0], pix_centroids[:, 1], pix_centroids[:, 2], s=600, c='black', marker='x')
            print(pix_centroids)
            plt.show()

        else:
            plotPixels(values, pix_centroids)

    # assign centroid of each contour or pixel
    for p in unique(preds):
        values[preds == p] = centroids[p]

    return values