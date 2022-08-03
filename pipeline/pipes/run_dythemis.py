from getFilterImageIDs import getFilterImageIDs
from inferImageClasses import inferImageClasses
from inferSegments import inferSegments
from extractHoneSegments import extractHoneSegments
from colorDiscretize import colorDiscretize
from inferClusterMorphs import inferClusterMorphs
from filterIDsWithComponents import filterIDsWithComponents
from getFilterImageIDs import getFilterImageIDs

# get ids not in train data
ids = getFilterImageIDs(records_fields=["genus"],records_values=["Dythemis"])

# infer classes
inferImageClasses(image_ids=ids, infer_colname="dorsal_lateral_bad", infer_names=["bad","dorsal","dorsolateral","lateral"],
                  model_name="4-class", print_steps=True)
ids = getFilterImageIDs(infer_fields=["dorsal_lateral_bad"],infer_values=["lateral"])

# 2_segmentations
# infer segment masks and save to data/masks
inferSegments(ids,model_name='segmenter_randaug6',increase_contrast=True,activation_threshold=0.5,print_steps=True,show=True)

# extract segments using masks, modifying the raw masks according to params
extractHoneSegments(ids,bound=True, remove_islands=False, set_nonwhite_to_black=True, erode=False,erode_kernel_size=0,print_steps=True,write=True)

ids = filterIDsWithComponents(ids,"segment")

# 3_discretization_compositing
colorDiscretize(ids,by_contours=False,print_details=True) #group_cluster_records_col="species")
inferClusterMorphs(ids,records_group_col="species",classes=["dorsal","lateral","dorsolateral"],cluster_image_type="pattern")
