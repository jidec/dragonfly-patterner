from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import io, data, img_as_float
import os
from skimage import data
from skimage.color import rgb2gray, rgb2hsv
import numpy as np
from skimage import util
from infillGlare import infillGlare
import cv2

img = cv2.imread("../../data/segments/INATRANDOM-24635495_segment.png")
infillGlare(img)

imgs = os.listdir("../../data/all_images")
imgs = imgs[1100:]
imgs = ["INATRANDOM-102083730.jpg","INAT-54529717-1.jpg","INAT-54622783-1.jpg","INATRANDOM-111036639.jpg"]
for img_name in imgs:
    og_img = io.imread("../../data/all_images" + "/" + img_name)

    plt.imshow(og_img)

    # create lum channel image
    lum_img = rgb2gray(og_img) #gray
    lum_img = img_as_float(lum_img)

    # create inverted sat channel image
    sat_img = rgb2hsv(og_img)
    sat_img = img_as_float(sat_img)
    sat_img = sat_img[:, :, 1]
    sat_img = util.invert(sat_img)

    # combine the two images
    new_img = (lum_img + sat_img) / 2
    plt.imshow(new_img)

    im = new_img
    # image_max is the dilation of im with a 20*20 structuring element
    # It is used within peak_local_max function
    image_max = ndi.maximum_filter(im, size=20, mode='constant') #'constant

    # Comparison between image_max and im to find the coordinates of local maxima
    coordinates = peak_local_max(im, min_distance=20) #20

    print(coordinates)

    # display results
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(im, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original')

    ax[1].imshow(image_max, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('Maximum filter')

    ax[2].imshow(im, cmap=plt.cm.gray)
    ax[2].autoscale(False)
    ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    ax[2].axis('off')
    ax[2].set_title('Peak local max')

    fig.tight_layout()
    plt.show()

    img = rgb2hsv(og_img)
    img = img[:, :, 1]

    im = img

    im = util.invert(im)

    # image_max is the dilation of im with a 20*20 structuring element
    # It is used within peak_local_max function
    image_max = ndi.maximum_filter(im, size=20, mode='constant')  # 'constant

    # Comparison between image_max and im to find the coordinates of local maxima
    coordinates_sat = peak_local_max(im, min_distance=20)  # 20

    # display results
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(im, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original')

    ax[1].imshow(image_max, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('Maximum filter')

    ax[2].imshow(im, cmap=plt.cm.gray)
    ax[2].autoscale(False)
    ax[2].plot(coordinates_sat[:, 1], coordinates_sat[:, 0], 'r.')
    ax[2].axis('off')
    ax[2].set_title('Peak local max')

    fig.tight_layout()

    plt.show()