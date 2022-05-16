from skimage.draw import disk
import numpy as np
import cv2
from PIL import Image

# return a vector of n CIELAB colors sampled from areas of random pixels in the numpy array color_segment
def sampleColorCircles(color_segment, n, radius):
    # convert image to CIELAB color space
    color_segment = cv2.cvtColor(color_segment, cv2.COLOR_BGR2LAB)
    # get only the pixels of the segment (exclude pure black background pixels)
    segment_pixels = np.any(color_segment != [0, 0, 0], axis=-1)

    # for each of n samples...
    for i in range(n):
        # pick a random pixel from segment pixels
        chosen_pixel = np.random.choice(segment_pixels,1)
        # get pixel coordinates within a circular range of the chosen pixel using skimage disk function
        # (or some other function that would just probably be easiest) https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.disk
        pixel_coordinates = disk(chosen_pixel, radius)
        # get only non-black pixels from these
        non_black_pixels = np.any(pixel_coordinates != [0,0,0])
        # calculate average CIE coordinate of those pixels and add it to a vector or list

    # return the vector


# usage of this function
# load image and convert to numpy array
img = Image.open("4155610_198_masked.png")
img = np.array(img, dtype=np.uint8)

# sample 10 colors
color_samples = sampleColorCircles(color_segment=img,n=10,radius=5)