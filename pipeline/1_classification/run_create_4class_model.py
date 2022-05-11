from sourceRdefs import moveSelectImages,mergeUpdateAnnotations,copyClassImagesForTraining,copyClassImagesToTrainingDirs
from loadTrainClassModel import loadTrainClassModel
from torchvision import transforms
from AddGaussianNoise import AddGaussianNoise
import shutil

# update annotations so all finished annotations from trainset_tasks are known
mergeUpdateAnnotations()

# specify size to reduce image to
image_pixels = 256
# set image augments
data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([image_pixels,image_pixels]),
            transforms.RandomHorizontalFlip(), # horizontally flip with 50% prob
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(0,360)),
            #transforms.GaussianBlur(kernel_size=7),
            #transforms.ColorJitter(brightness=(0.25,0.75),contrast=(0.25,0.75),saturation=(0.25,0.75)),
            transforms.ToTensor(), # convert a pil image or numpy ndarray to tensor
            #AddGaussianNoise(0., 1.), # add Gaussian noise
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # just normalization for validation
        'test': transforms.Compose([
            transforms.Resize([image_pixels,image_pixels]),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

# create dorsal_lateral_bad classifier model
# move dorsal images to dorsal dir, lateral images to lateral dir etc
copyClassImagesToTrainingDirs(class_col_name="dorsal_lateral_dorsolateral_bad",class_names=["dorsal","lateral","dorsolateral","bad"],ntest=12)

# load, train, and save model
# still have to update this to match new paradigm
loadTrainClassModel(data_dir="../../data/other/training_dirs/dorsal_lateral_dorsolateral_bad",
                    num_epochs=20, batch_size=6, num_workers=0,
                    data_transforms= data_transforms,
                    model_name="dorsal_lateral_bad",
                    model_dir="../../data/ml_models")

# remove copied training images from the temporary training dir
shutil.rmtree(path="../../data/training_dirs/dorsal_lateral_dorsolateral_bad")