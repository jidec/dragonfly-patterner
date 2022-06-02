from sourceRdefs import moveSelectImages,mergeUpdateAnnotations,copyClassImagesForTraining
from loadTrainClassModel import loadTrainClassModel
from torchvision import transforms
from AddGaussianNoise import AddGaussianNoise
import shutil

mergeUpdateAnnotations(skip_string="session")

image_pixels = 256
# set image augments
data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([256,256]),
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
            transforms.Resize([256,256]),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

class_col_name = "dorsal_lateral_dorsolateral_bad"

# create dorsal_lateral_bad classifier model
# move dorsal images to dorsal dir, lateral images to lateral dir etc
copyClassImagesForTraining(class_col = class_col_name, class_name = "dorsal",
                          to = "pylib/trainsets/" + class_col_name,
                          split_test_train = True, ntest = )
copyClassImagesForTraining(class_col = class_col_name, class_name = "lateral",
                          to = "pylib/trainsets/" + class_col_name,
                          split_test_train = True, ntest = 12)
copyClassImagesForTraining(class_col = class_col_name, class_name = "dorsolateral",
                          to = "pylib/trainsets/" + class_col_name,
                          split_test_train = True, ntest = 12)
copyClassImagesForTraining(class_col = class_col_name, class_name = "bad",
                          to = "pylib/trainsets/" + class_col_name,
                          split_test_train = True, ntest = 12)

# load, train, and save model
loadTrainClassModel(data_dir="pylib/trainsets/" + class_col_name,
                    num_epochs=20, batch_size=6, num_workers=0,
                    data_transforms= data_transforms,
                    model_name="dorsal_lateral_bad",
                    model_dir="pylib/models")

# remove training images
shutil.rmtree(path="pylib/trainsets/")