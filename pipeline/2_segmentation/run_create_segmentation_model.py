from loadTrainSegModel import loadTrainSegModel
from torchvision import transforms
from sourceRdefs import copyMaskImagesForTraining
from copyImagesToTrainingDir import copyImagesToTrainingDir
import os

# copy training segments from
mask_image_ids = os.listdir("../../data/masks/train_masks")
mask_image_ids = [s.removesuffix("_mask.jpg") for s in mask_image_ids]

copyImagesToTrainingDir("head_segmenter", mask_image_ids, 25, "mask", is_segment_training=True, proj_dir="../..")

data_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(224), # crop a random part of the image and resize it to size 224
        transforms.Resize([344, 344]),
        # transforms.Resize([448, 448]),
        # transforms.RandomHorizontalFlip(), # horizontally flip with 50% prob
        transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

loadTrainSegModel(data_dir="../../data/other/training_dirs/segmenter",
                  num_epochs=10, batch_size=4, num_workers=0,
                  data_transforms = data_transforms,
                  model_name="segmenter",
                  model_dir="../../data/ml_models", export_dir="../../data/ml_models")