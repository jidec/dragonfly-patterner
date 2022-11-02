from loadTrainSegModel import loadTrainSegModel
from torchvision import transforms
import torch

# contrast
data_transforms = transforms.Compose([
            transforms.Resize([344,344]),
            transforms.RandomAutocontrast(),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])

loadTrainSegModel(training_dir_name="segmenter_grey_vsmall",data_transforms=data_transforms,
                  num_epochs=13, batch_size=6, num_workers=0, resnet_size="50", pretrained=True,
                  criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(13)),
                  model_name="segmenter_grey_bce13_vsmall",proj_dir="../../..")

loadTrainSegModel(training_dir_name="segmenter_grey",data_transforms=data_transforms,
                  num_epochs=13, batch_size=6, num_workers=0, resnet_size="50", pretrained=True,
                  criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(13)),
                  model_name="segmenter_grey_bce13",proj_dir="../../..")

loadTrainSegModel(training_dir_name="segmenter_grey_small",data_transforms=data_transforms,
                  num_epochs=13, batch_size=6, num_workers=0, resnet_size="50", pretrained=True,
                  criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(13)),
                  model_name="segmenter_grey_bce13_small",proj_dir="../../..")

loadTrainSegModel(training_dir_name="segmenter_grey_big",data_transforms=data_transforms,
                  num_epochs=13, batch_size=6, num_workers=0, resnet_size="50", pretrained=True,
                  criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(13)),
                  model_name="segmenter_grey_bce13_big",proj_dir="../../..")
