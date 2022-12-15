from copyImagesToTrainingDir import copyImagesToTrainingDir
import os
from torchvision import transforms
from loadTrainSegModel import loadTrainSegModel
from extractHoneSegments import extractHoneSegments
from getFilterImageIDs import getFilterImageIDs
from inferSegments import inferSegments
import random

# copy training segments from
#mask_image_ids = os.listdir("../../data/other/train_masks")
#mask_image_ids = [s.removesuffix("_mask.jpg") for s in mask_image_ids]

#copyImagesToTrainingDir("segmenter", mask_image_ids, 75, "mask", is_segment_training=True, proj_dir="../..")

#data_transforms = transforms.Compose([
#            transforms.Resize([344, 344]),
#            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
#        ])
#loadTrainSegModel(training_dir_name="segmenter",data_transforms=data_transforms,
#                  num_epochs=13, batch_size=6, num_workers=0,
#                  model_name="segmenter_robadds")

#ids = getFilterImageIDs()
#random.shuffle(ids)
#ids = ids[0:9]
#inferSegments(image_ids=ids, model_name="segmenter_robadds",
              #greyscale=True, image_size=344, show=False, proj_dir="../..")

# contrast sharp
data_transforms = transforms.Compose([
            transforms.Resize([344,344]),
            transforms.RandomAutocontrast(),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])

loadTrainSegModel(training_dir_name="segmenter",data_transforms=data_transforms,
                  num_epochs=20, batch_size=6, num_workers=0, resnet_size="101",
                  model_name="segmenter_grey_contrast_sharp_b6_bce15_101_700")

#extractHoneSegments(ids,bound=True,remove_islands=True,set_nonwhite_to_black=True,erode_kernel_size=3,write=False,show=True,proj_dir="../..")
