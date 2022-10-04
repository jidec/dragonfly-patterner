from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import io, data, img_as_float
from skimage.color import rgb2gray, rgb2hsv
import numpy as np
from skimage import util
import cv2

def infillGlare(img,filter_size=20,min_peak_dist=20,sat_glare_range=20,show=False):
    og_img = img

    # create lum channel image
    lum_img = rgb2gray(og_img)
    lum_img = img_as_float(lum_img)

    # create inverted sat channel image
    sat_img = rgb2hsv(og_img)
    sat_img = img_as_float(sat_img)
    sat_img = sat_img[:, :, 1]
    sat_img = util.invert(sat_img)

    # combine the two images
    img = (lum_img + sat_img) / 2

    # image_max is the dilation of im with a 20*20 structuring element
    # it is used within peak_local_max function
    image_max = ndi.maximum_filter(img, size=filter_size, mode='constant')

    # Comparison between image_max and im to find the coordinates of local maxima
    coordinates = peak_local_max(img, min_distance=min_peak_dist)

    # display results of local max
    if show:
        fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
        ax = axes.ravel()
        ax[0].imshow(img, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('Original')

        ax[1].imshow(image_max, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('Maximum filter')
        ax[2].imshow(img, cmap=plt.cm.gray)
        ax[2].autoscale(False)
        ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
        ax[2].axis('off')
        ax[2].set_title('Peak local max')

        fig.tight_layout()
        plt.show()

    # convert to grey and threshold
    img_gray = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

    ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    blank = np.zeros_like(img)
    cv2.drawContours(blank, contours, -1, (255, 255, 255), -1)
    cv2.imshow("img", blank)
    cv2.waitKey(0)

    peak_contours = []
    for c in contours:
        cont_filled = np.zeros_like(img)
        cv2.drawContours(cont_filled, [c], -1, (255, 255, 255), thickness=-1)
        for xy in coordinates:
            if cont_filled[xy] == (255,255,255):
                peak_contours.append(c)

    peak_filled = np.zeros_like(img)
    cv2.drawContours(peak_filled, peak_contours, -1, (255, 255, 255), 1)
    cv2.imshow("img",peak_filled)
    cv2.waitKey(0)
