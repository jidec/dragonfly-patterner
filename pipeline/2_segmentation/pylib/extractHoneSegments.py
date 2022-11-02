from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
import cv2
import numpy as np
from showImages import showImages
from rotateToVertical import rotateToVertical
import warnings

def extractHoneSegments(image_ids,part_suffix=None,bound=True,rotate_to_vertical=True, remove_islands=True,erode_kernel_size=0,adj_to_background_col=False,target_bg_col=[160,160,160], target_range = 90,
                        set_nonwhite_to_black=False,write=True,show=False,print_steps=True,print_details=False,seg_subfolder="", proj_dir="../.."): #img_dir="../../data/all_images",mask_dir="../../data/masks"):

    """
        Hone masks, then extract and save their segments to data/segments

        :param List image_ids: the imageIDs (image names) to extract segments from
        :param str part_suffix: the part suffix for the masks to extract and infer from
        :param bool bound: bound segments to a bounding box
        :param bool rotate_to_vertical: rotate segments such that the central axis of the segment ellipsoid is vertical, with the heavier side of the segment on the top
        :param bool remove_islands: remove all but the largest island
        :param bool erode: erode the edges of segments -  use if model captures a bit of background at the edges of segments
        :param int erode_kernel_size: the size of the erosion kernel - every +1 adds significantly more erosion
        :param bool adj_to_background_grey: find the greyest color in the background and adjust images - used when images are taken of specimens on known grey cards
        :param bool write:

        :param bool show: whether or not to show image processing outputs and intermediates
        :param bool print_steps: whether or not to print processing step info after they are performed
        :param bool proj_dir: the path to the project directory containing /data and /trainset_tasks folders
    """
    image_ids = image_ids.copy()

    img_dir = proj_dir + "/data/all_images"
    mask_dir = proj_dir + "/data/masks"
    rgba_imgs_masks_ids = []

    if (print_steps): print("Started extractHoneSegments loop over images")
    if (print_steps): print("Estimated time " + str(1.5 * (len(image_ids) / 1000)) + " minutes")

    for index, id in enumerate(image_ids):
        # open image and convert to array
        img = cv2.imread(img_dir + "/" + id + ".jpg",cv2.IMREAD_COLOR)
        start_img = np.copy(img)
        if(print_details):{print("Processing ID " + id)}

        # open mask and convert to array
        mask = cv2.imread(mask_dir + "/" + id + "_mask.jpg",cv2.IMREAD_GRAYSCALE)

        if mask is None:
            continue
        if cv2.countNonZero(mask) == mask.shape[0] * mask.shape[1]:
            warnings.warn("Found empty mask, skipping it but consider filtering for ids with good/usable masks")
            continue

        # mask = cv2.bitwise_not(mask)
        start_mask = mask

        if set_nonwhite_to_black:
            mask[np.where(mask != 255)] = 0
            if (print_details): {print("Set nonwhite to black")}

        # erode border of mask
        if erode_kernel_size > 0:
            kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
            mask = cv2.erode(mask, kernel)
            if (print_details): {print("Eroded mask")}

        if remove_islands:
            # keep only biggest contour (i.e. remove islands from mask)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                # empty image and fill with big contour
                mask = np.zeros_like(mask)
                cv2.drawContours(mask, [max(contours, key=len)], -1, 255, thickness=-1)
                if (print_details): {print("Removed islands")}

        # apply mask to img to get masked img
        # showImages(show,[img,mask],["1","2"])

        bg_mask = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(img, img, mask=bg_mask)

        if (print_details): {print("Retrieved background")}

        if adj_to_background_col:

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

            if len(np.unique(preds)) > 1 and len(np.unique(cluster_values)) > 1:
                if (print_details): {print("Background pixel clustering successful")}
                clf.fit(cluster_values, preds)
                centers = clf.centroids_.astype(int)

                # find the closest color to the reference grey
                for index in range(0, centers.shape[0]):
                    row = centers[index]
                    if index == 0:
                        best_diff = 100000
                    diff = abs(row[0] - target_bg_col[0]) + abs(row[1] - target_bg_col[1]) + abs(row[2] - target_bg_col[2])
                    if diff < best_diff:
                        closest_to_grey = row
                        best_diff = diff

                if (print_details): {print("Found closest color to target: " + str(closest_to_grey))}

                # continue adjustment only if in range of reference grey and (maybe later) occupies a lot of size
                grey_dist = np.linalg.norm(closest_to_grey - np.array(target_bg_col)) #[160,160,160]
                if(grey_dist < target_range):
                    ref_b = closest_to_grey[0]
                    ref_g = closest_to_grey[1]
                    ref_r = closest_to_grey[2]

                    ref_lum = (ref_r + ref_g + ref_b) / 3

                    # if dark background, multiplier will be high, increasing the lightness of the image
                    # if light background, multiplier will be low, increasing the darkness of the image
                    img[:,:,0] = img[:,:,0] * ref_lum / ref_b
                    img[:,:,1] = img[:,:,1] * ref_lum / ref_g
                    img[:,:,2] = img[:,:,2] * ref_lum / ref_r

                    if (print_details): {print("Adjusted using known background color")}
                    showImages(show, [start_img,img], ["start","adj"])

        img = cv2.bitwise_and(img, img, mask=mask)
        if (print_details): {print("Applied mask to image")}

        # narrow image to bounding rect
        if bound:
            # create bounding rect around mask
            th = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            coords = cv2.findNonZero(th)
            x, y, w, h = cv2.boundingRect(coords)
            img = img[y:y + h, x:x + w]
            mask = mask[y:y + h, x:x + w]
            if (print_details): {print("Narrowed image to bounding box")}

        # Make a True/False mask of pixels whose BGR values sum to more than zero
        alpha = np.sum(img, axis=-1) > 0

        # Convert True/False to 0/255 and change type to "uint8" to match "na"
        alpha = np.uint8(alpha * 255)

        # Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
        img = np.dstack((img, alpha))

        if (print_details): {print("Added alpha channel")}

        if rotate_to_vertical:
            rotated = rotateToVertical([img],show=show)
            img = rotated[0]
            if (print_details): {print("Rotated to vertical")}

        # stop if image is 0 pixels in height (mask ended up nonexistent)
        if img.shape[0] != 0:
            # write segment
            if write:
                write_path = proj_dir + "/data/segments/" + seg_subfolder + id + "_segment.png"
                cv2.imwrite(write_path,img) # plus part suffix
                if (print_details): {print("Wrote segment to " + write_path)}

            # convert final masked image to RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

            showImages(show,[start_img,start_mask,mask,img],["Start Image","Start Mask","Adjusted Mask","Masked Segment"])

            # add new tuple to list
            rgba_imgs_masks_ids.append((img,mask,id))

        if index % 10 == 0:
            print("Processed " + str(index))

    return rgba_imgs_masks_ids