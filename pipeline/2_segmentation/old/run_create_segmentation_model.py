from loadTrainSegModel import loadTrainSegModel
from torchvision import transforms
from copyImagesToTrainingDir import copyImagesToTrainingDir
import os

# copy training segments from
#mask_image_ids = os.listdir("../../data/masks/train_masks")
#mask_image_ids = [s.removesuffix("_mask.jpg") for s in mask_image_ids]

#copyImagesToTrainingDir("segmenter", mask_image_ids, 30, "mask", is_segment_training=True, proj_dir="../..")

loadTrainSegModel(training_dir_name="segmenter",
                  num_epochs=20, batch_size=3, num_workers=0,
                  model_name="segmenter2")