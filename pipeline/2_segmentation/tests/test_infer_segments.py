from inferSegments import inferSegments
from getFilterImageIDs import getFilterImageIDs
import random
from extractHoneSegments import extractHoneSegments

image_ids = getFilterImageIDs(proj_dir="../../..",contains_str="RAND",train_fields=["has_segment","class"],train_values=[-1.0,"dorsal"])
random.shuffle(image_ids)
ids = image_ids[1:50]


inferSegments(image_ids=ids,model_name="segmenter_grey_contrast_sharp_b6_bce15_101",greyscale=True,image_size=344,show=False, proj_dir="../../..")#new_segmenter_greyrandaug
extractHoneSegments(ids,bound=True,rotate_to_vertical=True,remove_islands=False,
                    set_nonwhite_to_black=True,show=False,print_steps=False,write=True,proj_dir="../../..")
#segmenter_grey_b6_bce10_50
#segmenter_grey_sharp_b6_101