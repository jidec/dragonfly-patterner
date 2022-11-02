from getFilterImageIDs import getFilterImageIDs
from inferImageClasses import inferImageClasses
from extractHoneSegments import extractHoneSegments
from inferSegments import inferSegments
import random
from colorDiscretize import colorDiscretize

ids = getFilterImageIDs(not_in_train_data=True,records_fields=["is_aesh"],records_values=[True],proj_dir="../../..")
ids = random.sample(ids,5000)

colorDiscretize(ids,preclustered=True, group_cluster_raw_ids=True,write_subfolder="aesh_grouped5000", nclusters=4,
                scale=False,proj_dir="../../..",print_details=False)