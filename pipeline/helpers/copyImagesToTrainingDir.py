import random
import os
from copyImages import copyImages

def copyImagesToTrainingDir(training_dir_name, image_ids, ntest, class_dir_name, is_segment_training=False, seg_part_name=None, proj_dir="../.."):

    random.shuffle(image_ids)
    test_ids = image_ids[0:ntest-1]
    train_ids = image_ids[ntest:(len(image_ids) - 1)]

    training_dir = proj_dir + "/data/other/training_dirs/" + training_dir_name
    test_dir = training_dir + "/test"
    train_dir = training_dir + "/train"
    if not os.path.exists(training_dir):
        os.mkdir(training_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    if is_segment_training:
        # create train dirs
        os.mkdir(proj_dir + "/data/other/training_dirs/" + training_dir_name + "/train/image")
        os.mkdir(proj_dir + "/data/other/training_dirs/" + training_dir_name + "/train/mask")

        # create test dirs
        os.mkdir(proj_dir + "/data/other/training_dirs/" + training_dir_name + "/test/image")
        os.mkdir(proj_dir + "/data/other/training_dirs/" + training_dir_name + "/test/mask")

        if seg_part_name is not None:
            train_masks = [s + "_" + seg_part_name + "-mask" for s in train_ids]
            test_masks = [s + "_" + seg_part_name + "-mask" for s in test_ids]
        else:
            train_masks = [s + "_mask" for s in train_ids]
            test_masks = [s + "_mask" for s in test_ids]

        # copy train masks
        copyImages(proj_dir + "/data/masks/train_masks", training_dir + "/train/mask",train_masks)
        # copy train images
        copyImages(proj_dir + "/data/all_images", training_dir + "/train/image", train_ids)

        # copy test masks
        copyImages(proj_dir + "/data/masks/train_masks", training_dir + "/test/mask", test_masks)
        # copy test images
        copyImages(proj_dir + "/data/all_images", training_dir + "/test/image", test_ids)

    else:
        # create train dirs
        class_dir = proj_dir + "/data/other/training_dirs/" + training_dir_name + "/train/" + class_dir_name
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        # copy train images
        copyImages(proj_dir + "/data/all_images", class_dir, train_ids)

        # create test dirs
        class_dir = proj_dir + "/data/other/training_dirs/" + training_dir_name + "/test/" + class_dir_name
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        # copy test images
        copyImages(proj_dir + "/data/all_images", class_dir, test_ids)



