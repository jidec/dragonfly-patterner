from getFilterImageIDs import getFilterImageIDs
from extractHoneSegments import extractHoneSegments


ids = getFilterImageIDs(not_in_train_data=True,infer_fields=["bad_signifier"],infer_values=[0],
                        proj_dir="../../..")
ids = ids[2769:len(ids)]

extractHoneSegments(ids,bound=True,remove_islands=True,set_nonwhite_to_black=True,erode_kernel_size=3,write=True,show=False,proj_dir="../../..")

