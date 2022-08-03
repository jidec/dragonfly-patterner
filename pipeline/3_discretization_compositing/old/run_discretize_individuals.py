
from getFilterImageIDs import getFilterImageIDs
from extractSegments import extractSegments
from colorDiscretize import colorDiscretize


# get image IDs with segments
image_ids = getFilterImageIDs(only_segmented=True)
print(image_ids)
#image_ids = [str(x) for x in [1,2,3,6,8]]

# extract and refine segments
rgba_imgs_masks_names = extractSegments(image_ids,bound=True,remove_islands=True,erode=True,erode_kernel_size=3,write=False,show=True) #adj_to_background_grey=True

# discretize segments to patterns and save
colorDiscretize(rgba_imgs_masks_names, group_cluster=True, by_contours=True, min_contour_area=50, erode_contours=True, erode_kernel_size=3, cluster_model="kmeans",nclusters=4,cluster_min_samples=3,resize=True,show=False)

# consider removing outliers
