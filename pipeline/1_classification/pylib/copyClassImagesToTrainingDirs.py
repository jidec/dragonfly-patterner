import random
import os
from copyImages import copyImages

def copyImagesToTrainingDir(training_dir_name, image_ids, ntest, class_dir_name, is_segment_training, proj_dir="../.."):

    image_ids = random.shuffle(image_ids)
    test_ids = image_ids[0:ntest-1]
    train_ids = image_ids[ntest:(len(image_ids) - 1)]

    training_dir = proj_dir + "/data/other/training_dirs/" + training_dir_name
    os.mkdir(training_dir)

    os.mkdir(proj_dir + "/data/other/training_dirs/" + training_dir_name + "/test")
    os.mkdir(proj_dir + "/data/other/training_dirs/" + training_dir_name + "/train")

    if is_segment_training:
        # create train dirs
        os.mkdir(proj_dir + "/data/other/training_dirs/" + training_dir_name + "/train/image")
        os.mkdir(proj_dir + "/data/other/training_dirs/" + training_dir_name + "/train/mask")
        # copy train masks
        copyImages(proj_dir + "/data/masks/train_masks", training_dir + "/train/mask",train_ids + "_mask")
        # copy train images
        copyImages(proj_dir + "/data/all_images", training_dir + "/train/image", train_ids)

        # create test dirs
        os.mkdir(proj_dir + "/data/other/training_dirs/" + training_dir_name + "/test/image")
        os.mkdir(proj_dir + "/data/other/training_dirs/" + training_dir_name + "/test/mask")
        # copy test masks
        copyImages(proj_dir + "/data/masks/train_masks", training_dir + "/test/mask", train_ids + "_mask")
        # copy test images
        copyImages(proj_dir + "/data/all_images", training_dir + "/test/image", train_ids)

    else:
        # create train dirs
        class_dir = proj_dir + "/data/other/training_dirs/" + training_dir_name + "/train/" + class_dir_name
        os.mkdir(class_dir)
        # copy train images
        copyImages(proj_dir + "/data/all_images", class_dir, train_ids)

        # create test dirs
        class_dir = proj_dir + "/data/other/training_dirs/" + training_dir_name + "/test/" + class_dir_name
        os.mkdir(class_dir)
        # copy test images
        copyImages(proj_dir + "/data/all_images", class_dir, test_ids)



