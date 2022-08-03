from getFilterImageIDs import getFilterImageIDs
from extractSegments import extractSegments
from colorDiscretize import colorDiscretize


# get image IDs with segments
image_ids = getFilterImageIDs(train_fields=["has_segment"],train_values=[1.0])

# extract and refine segments
rgba_imgs_masks_names = extractSegments(image_ids,bound=True,remove_islands=True,erode=True,erode_kernel_size=3,rotate_to_vertical=True,write=True,show=True) #adj_to_background_grey=True