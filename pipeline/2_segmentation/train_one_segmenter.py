from extractHoneSegments import extractHoneSegments
from getFilterImageIDs import getFilterImageIDs
from inferSegments import inferSegments
from loadTrainSegModel import loadTrainSegModel
from torchvision import transforms
from copyImagesToTrainingDir import copyImagesToTrainingDir
from getFilterImageIDs import getFilterImageIDs
import torch

# NOTE = note string vs float distinction in values
#ids = getFilterImageIDs(train_fields=["has_segment"],train_values=[1.0])
#copyImagesToTrainingDir(training_dir_name="new_segmenter", image_ids=ids, ntest=30, is_segment_training=True, seg_part_name=None, proj_dir="../..")

data_transforms = transforms.Compose([
            transforms.Resize([344, 344]),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])
loadTrainSegModel(training_dir_name="new_segmenter_grey",data_transforms=data_transforms,
                  num_epochs=13, batch_size=6, num_workers=0,
                  model_name="new_segmenter_grey")
