from loadTrainClassModel import loadTrainClassModel
from torchvision import transforms
from AddGaussianNoise import AddGaussianNoise

data_dir = "dorsal_bad"
exp_dir = "out"
data_transforms = {
        # data augmentation and normalization for training
        # compose combines transforms
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(224), # crop a random part of the image and resize it to size 224
            transforms.Resize([400,400]),
            transforms.RandomHorizontalFlip(), # horizontally flip with 50% prob
            transforms.ToTensor(), # convert a pil image or numpy ndarray to tensor\
            AddGaussianNoise(0., 1.),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # just normalization for validation
        'test': transforms.Compose([
            transforms.Resize([400,400]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

# load and train
loadTrainClassModel(data_dir=data_dir, exp_dir=exp_dir, num_epochs=10, batch_size=4, num_workers=0,
                       data_transforms= data_transforms,model_name="dorsal_bad")