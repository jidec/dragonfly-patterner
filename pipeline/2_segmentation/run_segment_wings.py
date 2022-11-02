from mergePreprocessRecords import mergePreprocessRecords
from writeiNatGenusList import writeiNatGenusList
from getFilterImageIDs import getFilterImageIDs
from createOdomaticWingMasks import createOdomaticWingMasks
from colorDiscretize import colorDiscretize
import random

# for dragonfly-patterner, download Odonata research-grade iNat observations from USA
# https://www.gbif.org/occurrence/search?country=US&dataset_key=50c9509d-22c7-4a22-a47d-8c48425ef4a7&taxon_key=789
# place in root of data folder and rename as <preferred_dataset_name>_records.csv
# _records is a needed identifier for downstream functions

#mergePreprocessRecords(raw_records_csv_names=["names"],
#                       id_cols=["uniq_id"],id_prefixes=[""],csv_seps=[','],proj_root="D:/wing-color")

image_ids = getFilterImageIDs(proj_dir="D:/wing-color") #,infer_fields=["bad_signifier"],infer_values=[0])

#createOdomaticWingMasks(image_ids,proj_dir="D:/wing-color")

image_ids = [i + "_hind" for i in image_ids]
random.shuffle(image_ids)
image_ids = image_ids[1:1000]
colorDiscretize(image_ids,by_contours=False,show=False,group_cluster_raw_ids=True,cluster_model="kmeans",nclusters=5,print_details=True,proj_dir="D:/wing-color")