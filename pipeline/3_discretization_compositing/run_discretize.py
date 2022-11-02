from getFilterImageIDs import getFilterImageIDs
from colorDiscretize import colorDiscretize
from extractHoneSegments import extractHoneSegments
from filterIDsWithComponents import filterIDsWithComponents
from inferSegments import inferSegments
import random

# get image IDs with segments
#ids = getFilterImageIDs(train_fields=["has_segment"],train_values=[1.0])
#random.shuffle(ids)
ids = ["INATRANDOM-28343884","INATRANDOM-28645879"]

#inferSegments(image_ids=ids,model_name="segmenter_grey_sharp_b6_101",greyscale=True,image_size=344,show=True, proj_dir="../..")#new_segmenter_greyrandaug
colorDiscretize(ids,vert_resize=500,show=True,nclusters=5,cluster_eps=1.5,write_subfolder="individuals/", cluster_min_samples=30,cluster_model="gaussian_mixture",print_details=True)

# optics failing for some reason
