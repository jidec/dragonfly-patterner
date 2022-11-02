from getFilterImageIDs import getFilterImageIDs
from inferImageClasses import inferImageClasses
from extractHoneSegments import extractHoneSegments
from inferSegments import inferSegments
import random
from colorDiscretize import colorDiscretize

ids = getFilterImageIDs(not_in_train_data=True,records_fields=["is_gomph"],records_values=[True],
                        infer_fields=["conf_infers7"], infer_values= ["dorsal"], proj_dir="../../..")

colorDiscretize(ids,scale=True,use_positions=True,proj_dir="../../..",print_details=False,show=False)

colorDiscretize(ids,preclustered=True, group_cluster_raw_ids=True, write_subfolder="gomph_grouped_5000/", nclusters=5,
                scale=False,proj_dir="../../..",print_details=False)
