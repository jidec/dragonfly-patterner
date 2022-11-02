from getFilterImageIDs import getFilterImageIDs
from inferImageClasses import inferImageClasses
from extractHoneSegments import extractHoneSegments
from inferSegments import inferSegments
import random
from colorDiscretize import colorDiscretize

ids = getFilterImageIDs(not_in_train_data=True,records_fields=["is_gomph"],records_values=[True],proj_dir="../../..")
ids = random.sample(ids,5000)

inferImageClasses(image_ids=ids, infer_colname="dorsal_lateral_bad", infer_names= ["bad","dorsal","lateral"],
                  model_name="3-class-aug",show=False,proj_dir="../../..")

ids = getFilterImageIDs(not_in_train_data=True,records_fields=["is_gomph"],records_values=[True],
                        infer_fields=["conf_infers7"], infer_values= ["dorsal"], proj_dir="../../..")

inferSegments(image_ids=ids, model_name="segmenter_grey_contrast_sharp_b6_bce15_101",
              greyscale=True, image_size=344, show=False, activation_threshold=0.7,proj_dir="../../..")

extractHoneSegments(ids,bound=True,remove_islands=True,set_nonwhite_to_black=True,erode_kernel_size=5,write=True,show=False,proj_dir="../../..")

colorDiscretize(ids,scale=True,use_positions=True,proj_dir="../../..",print_details=False,show=False)

colorDiscretize(ids,preclustered=True, group_cluster_raw_ids=True,write_subfolder="gomph_grouped", nclusters=6,
                scale=False,proj_dir="../../..",print_details=False)