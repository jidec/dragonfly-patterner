from extractSegments import extractSegments
from getFilterImageIDs import getFilterImageIDs

# get ids with training segments
ids = getFilterImageIDs(train_fields=["has_segment"],train_values=[1])

# extract and write
extractSegments(ids, bound=True,remove_islands=True,erode=True,erode_kernel_size=4,rotate_to_vertical=True,write=True,show=False)