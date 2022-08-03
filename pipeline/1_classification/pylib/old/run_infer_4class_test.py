from inferImageClasses import inferImageClasses
from getFilterImageIDs import getFilterImageIDs
import pandas

## infer dorsal, lateral, or bad, then infer perfect dorsals and perfect laterals

# set dir to infer images from
image_dir = '../../../../data/all_images'

ids = getFilterImageIDs(not_in_train_data=True)
inferImageClasses(image_ids=ids, infer_colname= "dorsal_lateral_bad", infer_names= ["bad","dorsal","dorsolateral","lateral"],
                  model_location='../../data/ml_models/4-class.pt', image_size=344,show=False)
# combine inferences to infer dorsal, non-dorsal, or dorsal perfect
#for i in range(1,len(dorsal_lateral_bad_inference)):
#    if (dorsal_lateral_bad_inference[i] == "lateral") & (lateral_perfect_inference[i] == "perfect"):
#        dorsal_lateral_bad_inference[i] = "lateral_perfect"

