from inferSegments import inferSegments
from getFilterImageIDs import getFilterImageIDs
import random
from extractHoneSegments import extractHoneSegments
from rotateToVertical import rotateToVertical

image_ids = getFilterImageIDs(proj_dir="../../..",contains_str="RAND",train_fields=["has_segment"],train_values=[-1.0])


for id in image_ids:
    #inferSegments(activation_threshold=0.3,image_ids=[id],model_name="segmenter_grey_contrast_sharp_b6_bce15_101",greyscale=True,image_size=344,show=True, proj_dir="../../..")#new_segmenter_greyrandaug
    inferSegments(activation_threshold=0.7,image_ids=[id],model_name="segmenter_grey_contrast_sharp_b6_bce15_101",greyscale=True,image_size=344,show=False, proj_dir="../../..")#new_segmenter_greyrandaug
    #inferSegments(activation_threshold=0.7,increase_contrast=True,image_ids=[id],model_name="segmenter_grey_contrast_sharp_b6_bce15_101",greyscale=True,image_size=344,show=True, proj_dir="../../..")#new_segmenter_greyrandaug
    #inferSegments(activation_threshold=1,image_ids=[id],model_name="segmenter_grey_contrast_sharp_b6_bce15_101",greyscale=True,image_size=344,show=True, proj_dir="../../..")#new_segmenter_greyrandaug
    #inferSegments(activation_threshold=2,image_ids=[id],model_name="segmenter_grey_contrast_sharp_b6_bce15_101",greyscale=True,image_size=344,show=True, proj_dir="../../..")#new_segmenter_greyrandaug
    #inferSegments(activation_threshold=0.40,image_ids=[id],model_name="segmenter_grey_contrast_sharp_b6_bce15_101",greyscale=True,image_size=344,show=True, proj_dir="../../..")#new_segmenter_greyrandaug

    #inferSegments(image_ids=[id],model_name="segmenter_grey_bce20",greyscale=True,image_size=344,show=True, proj_dir="../../..")#new_segmenter_greyrandaug
    #inferSegments(image_ids=[id],model_name="segmenter_grey_bce10",greyscale=True,image_size=344,show=True, proj_dir="../../..")#new_segmenter_greyrandaug

# smaller the bce makes segs fit more
#extractHoneSegments(ids,bound=True,rotate_to_vertical=True,remove_islands=False,
#                    set_nonwhite_to_black=True,show=False,print_steps=False,write=True,proj_dir="../../..")
#segmenter_grey_b6_bce10_50
#segmenter_grey_sharp_b6_101