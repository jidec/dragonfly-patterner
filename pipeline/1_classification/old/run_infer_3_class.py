from inferImageClasses import inferImageClasses
from sourceRdefs import getFilterImageIDs

# get image IDs not in training set, not already classified
image_ids = getFilterImageIDs(exclude_training=False)

# infer classes of images and save results in inferences.csv in data folder
inferImageClasses(image_ids=image_ids, infer_colname= "dorsal_lateral_bad", infer_names= ["bad","dorsal","lateral"],
                  model_location='pylib/models/dorsal_lateral_bad.pt', image_size=400,show=True)