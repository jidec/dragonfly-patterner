from inferImageClasses import inferImageClasses
from getFilterImageIDs import getFilterImageIDs

ids = getFilterImageIDs(not_in_train_data=True)

ids = ids[11:20]

inferImageClasses(image_ids=ids, infer_colname="dorsal_lateral_bad", infer_names=["bad","dorsal","dorsolateral","lateral"],
                  model_name="4-class", print_steps=True, image_size=344,show=False)