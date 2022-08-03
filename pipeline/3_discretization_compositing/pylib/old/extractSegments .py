from PIL import Image
from PIL import ImageOps
import numpy as np
import cv2
from skimage import measure
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
import cv2
import numpy as np
from scipy.ndimage.filters import minimum_filter
from showImages import showImages
from numpy import unique
import colorsys
from rotateToVertical import rotateToVertical

# extractSegment returns a list of rgba images, masks, and ids tuples to be input to a discretize function
def extractHoneSegments(image_ids,bound=True,remove_islands=True,erode=True,remove_glare=False,adj_to_background=False,adj_to_background_grey=False,background_mean=None,rotate_to_vertical=False,erode_kernel_size=4,write=True,show=False,proj_dir="../../"): #img_dir="../../data/all_images",mask_dir="../../data/masks"):

    """
        Hone masks, then extract and save their segments

            :param List image_ids: the imageIDs (image names) to infer from
            :param str model_name: the name of the model contained in data/ml_models to infer with NOT including the .pt suffix
            :param int image_size: the image dimensions that the model was trained on
            :param str part_suffix:
            :param float activation_threshold: the amount the neuron must be activated to register a pixel as part of the segment
                This should be fine-tuned for each use case
            :param bool show: whether or not to show image processing outputs and intermediates
            :param bool print_steps: whether or not to print processing step info after they are performed
        """
    img_dir  = proj_dir + "/data/all_images"
    mask_dir = proj_dir + "/data/masks"
    rgba_imgs_masks_ids = []

    # if adjusting to background and background mean not specified, calculate it
    # not great, pretty redundant, move somewhere else when finalizing this method
    # probably will involve separating out the mask honing as another method
    if adj_to_background and background_mean == None:
        bg_colors = []
        for id in image_ids:
            img = cv2.imread(img_dir + "/" + id + ".jpg", cv2.IMREAD_COLOR)
            # open mask and convert to array
            mask = cv2.imread(mask_dir + "/" + id + "_mask.jpg", cv2.IMREAD_GRAYSCALE)
            mask = cv2.bitwise_not(mask)

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
            background = cv2.bitwise_not(img, img, mask=mask)
            bg_colors.append(background.mean(axis=(0,1)))
        background_mean = sum(bg_colors) / 3

    for id in image_ids:
        # open image and convert to array
        img = cv2.imread(img_dir + "/" + id + ".jpg",cv2.IMREAD_COLOR)
        start_img = np.copy(img)

        print("ID:" + id)

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
        showImages(show,[img,mask],["1","2"])

        bg_mask = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(img, img, mask=bg_mask)

        if adj_to_background:
            background = cv2.bitwise_not(img,img,mask=mask)
            bg_rgb = background.mean(axis=(0,1))
            bg_luminance = sum(bg_rgb) / 3
            bg_lum_mult = background_mean / bg_luminance
            #if dark background, multiplier will be high, increasing the lightness of the image
            #if light background, multiplier will be low, increasing the darkness of the image
            img *= bg_lum_mult
            # adjust the masked image using this difference
            # change value of red channel
            #img[:, :, 1] =

        if adj_to_background_grey:

            showImages(show, [bg], ["bg"])

            # save initial image shape
            sh = np.shape(img)
            width = np.shape(img)[0]
            length = np.shape(img)[1]

            # get pixels in bg
            bg_pixels = []
            for i in range(0, width):
                for j in range(0, length):
                    # only add non-black pixels
                    l = bg[i,j].tolist()
                    if(l != [0,0,0]):
                        bg_pixels.append(bg[i,j])
            # print(bg_pixels)
            cluster_values = np.array(bg_pixels)
            model = KMeans(n_clusters=3)
            model.fit(cluster_values)
            # get the cluster with the lowest saturation
            preds = model.predict(cluster_values)
            clf = NearestCentroid()
            clf.fit(cluster_values, preds)
            centers = clf.centroids_.astype(int)
            print(centers)


            #for index in range(0,centers.shape[0]):
            #    row = centers[index]
            #    if index == 0:
            #        least_sat_hls = colorsys.rgb_to_hls(row[2]/255,row[1]/255,row[0]/255)
            #    hls = colorsys.rgb_to_hls(row[2]/255,row[1]/255,row[0]/255)
            #    if hls[2] < least_sat_hls[2]:
            #        least_sat_hls = hls
            #    #print(hls[hls[:, 2].argmin()])
            #least_sat_rgb = colorsys.hls_to_rgb(least_sat_hls[0],least_sat_hls[1],least_sat_hls[2])
            #least_sat_rgb = [int(r * 255) for r in least_sat_rgb]

            # find the closest color to the reference grey
            for index in range(0, centers.shape[0]):
                row = centers[index]
                if index == 0:
                    best_diff = 100000
                diff = abs(row[0] - 160) + abs(row[1] - 160) + abs(row[2] - 160)
                print(diff)
                if diff < best_diff:
                    closest_to_grey = row
                    best_diff = diff

            print(closest_to_grey)

            # continue adjustment only if in range of reference grey and (maybe later) occupies a lot of size
            grey_dist = np.linalg.norm(closest_to_grey - np.array([160, 160, 160]))
            if(grey_dist < 60):
                ref_b = closest_to_grey[0]
                ref_g = closest_to_grey[1]
                ref_r = closest_to_grey[2]

                ref_lum = (ref_r + ref_g + ref_b) / 3

                # if dark background, multiplier will be high, increasing the lightness of the image
                # if light background, multiplier will be low, increasing the darkness of the image
                img[:,:,0] = img[:,:,0] * ref_lum / ref_b
                img[:,:,1] = img[:,:,1] * ref_lum / ref_g
                img[:,:,2] = img[:,:,2] * ref_lum / ref_r

                showImages(show, [start_img,img], ["start","adj"])
                # adjust the masked image using this difference
                # change value of red channel
                # img[:, :, 1] =

        img = cv2.bitwise_and(img, img, mask=mask)

        # create bounding rect around mask
        th = cv2.threshold(mask, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        coords = cv2.findNonZero(th)
        x, y, w, h = cv2.boundingRect(coords)

        # narrow image to bounding rect
        if bound:
            img = img[y:y + h, x:x + w]
            mask = mask[y:y + h, x:x + w]

        # Make a True/False mask of pixels whose BGR values sum to more than zero
        alpha = np.sum(img, axis=-1) > 0

        # Convert True/False to 0/255 and change type to "uint8" to match "na"
        alpha = np.uint8(alpha * 255)

        # Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
        img = np.dstack((img, alpha))

        if rotate_to_vertical:
            rotated = rotateToVertical([img],show=show)
            img = rotated[0]

        # write segment
        if write:
            cv2.imwrite("../../data/segments/" + id + "_segment.png",img)

        # convert final masked image to RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

        showImages(show,[start_img,start_mask,mask,img],["Start Image","Start Mask","Adjusted Mask","Masked Segment"])

        # add new tuple to list
        rgba_imgs_masks_ids.append((img,mask,id))

    return rgba_imgs_masks_ids