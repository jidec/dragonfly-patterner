from sourceRdefs import getFilterImageIDs
from extractSegments import extractSegments
from discretizeIndvs import discretizeIndvs

# get image IDs with segments
image_ids = getFilterImageIDs(only_segmented=True)
image_ids = image_ids[3:7]
#image_ids = [image_ids[1]]
#print(image_ids)
#image_ids = ["1088267_677"] #,"31364710_443"]
#image_ids = ["31364710_443"]

# discretize
rgba_imgs_masks_names = extractSegments(image_ids,bound=True,remove_islands=True,erode=True,erode_kernel_size=4,remove_glare=False,show=False,mask_dir="../../data/masks/train_masks")

# discretize individual by individual, save to data and align vertically for good measure
discretizeIndvs(rgba_imgs_masks_names, by_contours=True, min_contour_area=10, cluster_model="optics", nclusters=5,show=True)

# consider removing outliers
