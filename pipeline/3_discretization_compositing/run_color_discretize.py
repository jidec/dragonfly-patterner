from getFilterImageIDs import getFilterImageIDs
from colorDiscretize import colorDiscretize
from extractHoneSegments import extractHoneSegments
from filterIDsWithComponents import filterIDsWithComponents
from inferSegments import inferSegments
import random

# get image IDs with segments
ids = getFilterImageIDs(train_fields=["has_segment"],train_values=[1.0])
#ids = filterIDsWithComponents(ids,"segment")
random.shuffle(ids)
ids = ids[1:50]
# ids = ["INATRANDOM-32614488","INATRANDOM-31557604","INATRANDOM-8364450","INATRANDOM-17396628","INATRANDOM-42042128","INATRANDOM-51324309"]
#ids = ["INAT-18294808-1"]
#inferSegments(image_ids=ids,model_name="segmenter_grey_sharp_b6_101",greyscale=True,image_size=344,show=True, proj_dir="../..")#new_segmenter_greyrandaug
extractHoneSegments(ids,bound=True,rotate_to_vertical=True,remove_islands=False,
                    set_nonwhite_to_black=True,show=False,print_steps=False,write=True,proj_dir="../..")
colorDiscretize(ids,by_contours=False,show=False,group_cluster_raw_ids=False,cluster_model="kmeans",nclusters=3,print_details=True) #group_cluster_records_col="species")