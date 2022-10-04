from loadTrainSegModel import loadTrainSegModel
from torchvision import transforms
import torch

# contrast
data_transforms = transforms.Compose([
            transforms.Resize([344,344]),
            transforms.RandomAutocontrast(),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])
loadTrainSegModel(training_dir_name="new_segmenter_grey",data_transforms=data_transforms,
                  num_epochs=20, batch_size=6, num_workers=0, resnet_size="101",
                  model_name="segmenter_grey_contrast_b6_bce15_101")

# contrast sharp
data_transforms = transforms.Compose([
            transforms.Resize([344,344]),
            transforms.RandomAutocontrast(),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])
loadTrainSegModel(training_dir_name="new_segmenter_grey",data_transforms=data_transforms,
                  num_epochs=20, batch_size=6, num_workers=0, resnet_size="101",
                  model_name="segmenter_grey_contrast_sharp_b6_bce15_101")

# blur
data_transforms = transforms.Compose([
            transforms.Resize([344,344]),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])
loadTrainSegModel(training_dir_name="new_segmenter_grey",data_transforms=data_transforms,
                  num_epochs=20, batch_size=6, num_workers=0, resnet_size="101",
                  model_name="segmenter_grey_blur_b6_bce15_101")
input("")

# persp
data_transforms = transforms.Compose([
            transforms.Resize([344,344]),
            transforms.RandomPerspective(distortion_scale=0.5, p=1.0),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])
loadTrainSegModel(training_dir_name="new_segmenter_grey",data_transforms=data_transforms,
                  num_epochs=20, batch_size=6, num_workers=0, resnet_size="101",
                  model_name="segmenter_grey_persp_b6_101")
# sharp
data_transforms = transforms.Compose([
            transforms.Resize([344,344]),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])
loadTrainSegModel(training_dir_name="new_segmenter_grey",data_transforms=data_transforms,
                  num_epochs=20, batch_size=6, num_workers=0, resnet_size="101",
                  model_name="segmenter_grey_jitter_b6_101")

# sharp
data_transforms = transforms.Compose([
            transforms.Resize([344,344]),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])
loadTrainSegModel(training_dir_name="new_segmenter_grey",data_transforms=data_transforms,
                  num_epochs=20, batch_size=6, num_workers=0, resnet_size="101",
                  model_name="segmenter_grey_sharp_b6_101")
