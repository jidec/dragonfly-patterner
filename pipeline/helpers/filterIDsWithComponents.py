import os

def filterIDsWithComponents(image_ids,component_type="segment",proj_dir="../.."):
    if component_type == "segment":
        files = os.listdir(proj_dir + "/data/segments")
    elif component_type == "mask":
        files = os.listdir(proj_dir + "/data/masks")

    component_ids = [i.split("_")[0] for i in files]
    ids = set(image_ids).intersection(set(component_ids))
    return(list(ids))
