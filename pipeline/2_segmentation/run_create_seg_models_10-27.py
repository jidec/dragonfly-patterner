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
loadTrainSegModel(training_dir_name="new_segmenter_grey",data_transforms=data_transforms,
                  num_epochs=13, batch_size=6, num_workers=0, resnet_size="50", pretrained=True,
                  criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10)),
                  model_name="segmenter_grey_bce10")

loadTrainSegModel(training_dir_name="new_segmenter_grey",data_transforms=data_transforms,
                  num_epochs=13, batch_size=6, num_workers=0, resnet_size="50", pretrained=True,
                  criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20)),
                  model_name="segmenter_grey_bce20")
