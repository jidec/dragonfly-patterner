from downloadiNatGenusImages import downloadiNatGenusImages
from getFilterImageIDs import getFilterImageIDs
from inferImageClasses import inferImageClasses
from extractHoneSegments import extractHoneSegments

gomph_indices = [50, 55,  9, 59, 23, 24, 28, 83, 18, 14, 47, 42, 63, 57, 91, 62, 73, 85]
gomph_indices = [85]
for i in gomph_indices:
    downloadiNatGenusImages(start_index=i,end_index=i+1,proj_dir="../../..")

ids = getFilterImageIDs(not_in_train_data=True,records_fields=["family"],records_values=["Gomphidae"],proj_dir="../../..")

inferImageClasses(image_ids=ids, infer_colname="dorsal_lateral_bad", infer_names= ["bad","dorsal","lateral"],
                  model_name="3-class-aug",show=False)

extractHoneSegments(ids,bound=True,remove_islands=True,set_nonwhite_to_black=True,erode_kernel_size=3,write=True,show=False,proj_dir="../../..")
