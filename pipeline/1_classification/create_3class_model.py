from sourceRdefs import moveSelectImages,mergeUpdateAnnotations,copyClassImagesForTraining
from loadTrainClassModel import loadTrainClassModel
from torchvision import transforms
from AddGaussianNoise import AddGaussianNoise
import shutil

# set image augments
data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([400,400]),
            transforms.RandomHorizontalFlip(), # horizontally flip with 50% prob
            transforms.ToTensor(), # convert a pil image or numpy ndarray to tensor
            AddGaussianNoise(0., 1.), # add Gaussian noise
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # just normalization for validation
        'test': transforms.Compose([
            transforms.Resize([400,400]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

# create dorsal_lateral_bad classifier model
# move dorsal images to dorsal dir, lateral images to lateral dir etc
copyClassImagesForTraining(class_col = "dorsal_lateral_bad", class_name = "dorsal",
                          to = "pylib/trainsets/dorsal_lateral_bad",
                          split_test_train = True, ntest = 5)
copyClassImagesForTraining(class_col = "dorsal_lateral_bad", class_name = "lateral",
                          to = "pylib/trainsets/dorsal_lateral_bad",
                          split_test_train = True, ntest = 5)
copyClassImagesForTraining(class_col = "dorsal_lateral_bad", class_name = "bad",
                          to = "pylib/trainsets/dorsal_lateral_bad",
                          split_test_train = True, ntest = 5)

# load, train, and save model
loadTrainClassModel(data_dir="pylib/trainsets/dorsal_lateral_bad",
                    num_epochs=10, batch_size=4, num_workers=0,
                    data_transforms= data_transforms,
                    model_name="dorsal_lateral_bad",
                    model_dir="pylib/models")

# remove training images
shutil.rmtree(path="pylib/trainsets/")