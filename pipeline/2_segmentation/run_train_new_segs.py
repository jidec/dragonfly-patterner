from loadTrainSegModel import loadTrainSegModel
from torchvision import transforms
import torch

# base grey segmenter
data_transforms = transforms.Compose([
            transforms.Resize([344,344]),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])
loadTrainSegModel(training_dir_name="new_segmenter_grey",data_transforms=data_transforms,
                  num_epochs=20, batch_size=6, num_workers=0, resnet_size="50",
                  model_name="segmenter_grey_b6_50")

# b6 101
data_transforms = transforms.Compose([
            transforms.Resize([344,344]),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])
loadTrainSegModel(training_dir_name="new_segmenter_grey",data_transforms=data_transforms,
                  num_epochs=20, batch_size=6, num_workers=0, resnet_size="101",
                  model_name="segmenter_grey_b6_101")

# bce 10
data_transforms = transforms.Compose([
            transforms.Resize([344,344]),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])
loadTrainSegModel(training_dir_name="new_segmenter_grey",data_transforms=data_transforms,
                  num_epochs=20, batch_size=6, num_workers=0, resnet_size="50", criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10)),
                  model_name="segmenter_grey_b6_bce10_50")

# flip
data_transforms = transforms.Compose([
            transforms.Resize([344,344]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])
loadTrainSegModel(training_dir_name="new_segmenter_grey",data_transforms=data_transforms,
                  num_epochs=20, batch_size=6, num_workers=0, resnet_size="101",
                  model_name="segmenter_grey_flip_b6_101")

# sharp
data_transforms = transforms.Compose([
            transforms.Resize([344,344]),
            transforms.RandomAdjustSharpness(),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])
loadTrainSegModel(training_dir_name="new_segmenter_grey",data_transforms=data_transforms,
                  num_epochs=20, batch_size=6, num_workers=0, resnet_size="101",
                  model_name="segmenter_grey_sharp_b6_101")