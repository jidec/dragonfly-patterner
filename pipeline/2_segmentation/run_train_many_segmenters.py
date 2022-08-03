from torchvision import transforms
from loadTrainSegModel import loadTrainSegModel

data_transforms = transforms.Compose([
            transforms.Resize([344, 344]),
            transforms.RandAugment(),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])
loadTrainSegModel(training_dir_name="segmenter",data_transforms=None,
                  num_epochs=13, batch_size=6, num_workers=0,
                  model_name="segmenter_randaug6")

data_transforms = transforms.Compose([
            transforms.Resize([344, 344]),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])
loadTrainSegModel(training_dir_name="segmenter",data_transforms=None,
                  num_epochs=13, batch_size=6, num_workers=0,
                  model_name="segmenter6")

data_transforms = transforms.Compose([
            transforms.Resize([344, 344]),
            transforms.Grayscale(),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])
loadTrainSegModel(training_dir_name="segmenter",data_transforms=None,
                  num_epochs=13, batch_size=6, num_workers=0,
                  model_name="segmenter_grey6")

data_transforms = transforms.Compose([
            transforms.Resize([344, 344]),
            transforms.RandomHorizontalFlip(), # horizontally flip with 50% prob
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
            transforms.RandomAutocontrast(),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        ])
loadTrainSegModel(training_dir_name="segmenter",data_transforms=None,
                  num_epochs=13, batch_size=6, num_workers=0,
                  model_name="segmenter_augs6")