from loadTrainSegModel import loadTrainSegModel
from torchvision import transforms
from sourceRdefs import copyMaskImagesForTraining

# copy training segments from
#copyMaskImagesForTraining(from_="G:/ant-patterner/data/masks/train_masks", imgs_from="G:/ant-patterner/data/all_images",
#                          to="G:/ant-patterner/data/other/training_dirs", ntest=10)

data_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(224), # crop a random part of the image and resize it to size 224
        transforms.Resize([200, 200]), #344
        # transforms.Resize([448, 448]),
        # transforms.RandomHorizontalFlip(), # horizontally flip with 50% prob
        transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

loadTrainSegModel(data_dir="E:/ant-patterner/data/other/training_dirs/thorax",
                  num_epochs=10, batch_size=7, num_workers=0,
                  data_transforms = data_transforms,
                  model_name="ant_thorax_segment",
                  model_dir="E:/ant-patterner/data/ml_models",
                  export_dir="E:/ant-patterner/data")