from getFilterImageIDs import getFilterImageIDs
from inferClusterMorphs import inferClusterMorphs
from colorDiscretize import colorDiscretize

# get IDs of a genus
ids = getFilterImageIDs(records_fields=["genus"],records_values=["Dythemis"])

#colorDiscretize(ids,

# cluster for morphs for each species-view combination and add to them to data/inferences.csv
inferClusterMorphs(ids,records_group_col="species",classes=["dorsal","lateral","dorsolateral"],cluster_image_type="pattern")