from getFilterImageIDs import getFilterImageIDs
from colorDiscretize import colorDiscretize
from extractHoneSegments import extractHoneSegments
from filterIDsWithComponents import filterIDsWithComponents

# get image IDs with segments
#ids = getFilterImageIDs(train_fields=["has_segment"],train_values=[1.0])
#ids = filterIDsWithComponents(ids,"segment")
#ids = ids[5:10]
ids = ["INATRANDOM-32614488","INATRANDOM-31557604","INATRANDOM-8364450","INATRANDOM-17396628","INATRANDOM-42042128","INATRANDOM-51324309"]
colorDiscretize(ids,by_contours=False,show=False,group_cluster_raw_ids=True,cluster_model="optics",nclusters=4,print_details=True) #group_cluster_records_col="species")