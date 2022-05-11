from PIL import Image
from PIL import ImageOps
import numpy as np
import cv2
from skimage import measure
import cv2
import numpy as np
from scipy.ndimage.filters import minimum_filter
from helpers import showImages

# extractSegment returns a list of rgba images, masks, and ids tuples to be input to a discretize function
def extractSegments(image_ids,bound=True,remove_islands=True,erode=True,remove_glare=False,erode_kernel_size=4,write=True,show=False,img_dir="../../data/all_images",mask_dir="../../data/masks"):

    rgba_imgs_masks_ids = []
    for id in image_ids:
        # open image and convert to array
        img = cv2.imread(img_dir + "/" + id + ".jpg",cv2.IMREAD_COLOR)
        start_img = np.copy(img)

        if remove_glare:
            # step 1: compute the dark channel
            def dark_channel(im, patch_size=15):
                dark = minimum_filter(im, patch_size, mode='nearest')
                dark = np.min(dark, axis=2)
                return dark

            def atmospheric_light(im, dark, mean=True):
                # We first pick the top 0.1% brightest pixels in the dark channel.
                # Among these pixels, the pixels with highest intensity in the input
                # image I is selected as the atmospheric light
                flat = dark.flatten()
                num = flat.shape[0] >> 10  # same as / 1024
                assert num >= 1
                indice = flat.argsort()[-num:]
                cols = dark.shape[1]
                xys = [(index // cols, index % cols) for index in indice]
                # In paper, author haven't say we should use average
                # but in practice, average value yield better result
                if mean:
                    points = np.array([im[xy] for xy in xys])
                    airlight = points.mean(axis=0)
                    return airlight
                xys = sorted(xys, key=lambda xy: sum(im[xy]), reverse=True)
                xy = xys[0]
                airlight = im[xy]
                return airlight

            def estimate_transmission(im, airlight, patch_size=15):
                normal = im / airlight
                tx = 1 - 0.95 * dark_channel(normal, patch_size)
                return tx

            def recover_scene(im, airlight, tx, t0=0.1):
                mtx = np.where(tx > t0, tx, t0)
                res = np.zeros_like(im, dtype=im.dtype)
                for i in range(3):
                    c = (im[:, :, i] - airlight[i]) / mtx + airlight[i]
                    c = np.where(c < 0, 0, c)
                    c = np.where(c > 255, 255, c)
                    res[:, :, i] = c
                return res

            im = np.copy(img)
            patch_size = 10
            dark = dark_channel(im, patch_size)
            dark_flat = dark.flatten()
            im = im.reshape(im.shape[0]*im.shape[1], im.shape[2])
            for i in range(0,len(im)):
                im[i] = im[i] - (dark_flat[i] * 1)
            im[im<=0] = 0
            im = im.reshape(img.shape)
            showImages(show,[img,dark,im],["Image","Dark Map","Unglared Image"])
            img = im

        # open mask and convert to array
        mask = cv2.imread(mask_dir + "/" + id + "_mask.jpg",cv2.IMREAD_GRAYSCALE)
        mask = cv2.bitwise_not(mask)
        start_mask = mask

        # erode border of mask
        if erode:
            kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
            mask = cv2.erode(mask, kernel)

        if remove_islands:
            # keep only biggest contour (i.e. remove islands from mask)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # empty image and fill with big contour
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [max(contours, key=len)], -1, 255, thickness=-1)

        # apply mask to img to get masked img
        # showImages(show,[img,mask],["1","2"])
        img = cv2.bitwise_and(img, img, mask=mask)

        # create bounding rect around mask
        th = cv2.threshold(mask, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        coords = cv2.findNonZero(th)
        x, y, w, h = cv2.boundingRect(coords)

        # narrow image to bounding rect
        if bound:
            img = img[y:y + h, x:x + w]
            mask = mask[y:y + h, x:x + w]

        # write segment
        if write:
            cv2.imwrite("../../data/segments/" + id + "_segment.jpg",img)

        # convert final masked image to RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

        showImages(show,[start_img,start_mask,mask,img],["Start Image","Start Mask","Adjusted Mask","Masked Segment"])

        # add new tuple to list
        rgba_imgs_masks_ids.append((img,mask,id))

    return rgba_imgs_masks_ids