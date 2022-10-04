from loadTrainSegModel import loadTrainSegModel
from torchvision import transforms
from copyImagesToTrainingDir import copyImagesToTrainingDir
import torch
from getFilterImageIDs import getFilterImageIDs

# copy training segments from

data_transforms = transforms.Compose([
        transforms.Resize([200, 200]), #344
        transforms.RandomHorizontalFlip(), # horizontally flip with 50% prob
        transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
    ])

loadTrainSegModel(training_dir_name="head",
                  num_epochs=10, batch_size=7, num_workers=0,
                  data_transforms = data_transforms,
                  model_name="ant_head_segment2",
                  proj_dir="D:/ant-patterner",
                  criterion = torch.nn.BCEWithLogitsLoss())
