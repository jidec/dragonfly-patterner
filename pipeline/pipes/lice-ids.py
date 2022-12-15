#from sourceRdefs import moveSelectImages,mergeUpdateAnnotations,copyClassImagesForTraining,copyClassImagesToTrainingDirs
from updateTrainingMetadata import updateTrainingMetadata
from loadTrainClassModel import loadTrainClassModel
from torchvision import transforms
from AddGaussianNoise import AddGaussianNoise
import shutil
from getFilterImageIDs import getFilterImageIDs
from copyImagesToTrainingDir import copyImagesToTrainingDir

# 87% lateral #90% dorsal # 94% dl

# update annotations so all finished annotations from trainset_tasks are known
#updateTrainingMetadata()

# specify size to reduce image to
image_pixels = 344

# set image augments
data_transforms = {
        'default': transforms.Compose([
            transforms.Resize([image_pixels,image_pixels]),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'train': transforms.Compose([
            transforms.Resize([image_pixels,image_pixels]),
            transforms.RandomAutocontrast(),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomHorizontalFlip(1), # horizontally flip with 50% prob
            transforms.ToTensor(), # convert a pil image or numpy ndarray to tensor
        ]),
    }


# load, train, and save model
# still have to update this to match new paradigm
loadTrainClassModel(training_dir_name="lice_males_ventral",
                    num_epochs=20, batch_size=6, num_workers=0,
                    data_transforms= data_transforms,
                    model_name="lice_males_ventral",proj_dir="E:/lice-ids")

# remove copied training images from the temporary training dir
# shutil.rmtree(path="../../data/training_dirs/4-class")