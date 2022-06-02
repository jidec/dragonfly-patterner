import rpy2.robjects as ro
import os
import rpy2.robjects as robj

def preprocessiNat(proj_root="../.."):
    """
        Preprocess iNat data by adding numImages column and removing extraneous columns
        :param str proj_root: the location of the project folder i.e. dragonfly-patterner containing an /R/src/preprocessiNat.R script to source from
    """
    r = ro.r
    r.source(proj_root + "/R/src/preprocessiNat.R")
    r.preprocessiNat(proj_root)

def writeGenusList(proj_root="../.."):
    """
        Write a list of genera to data folder for use by downloader scripts
    """
    r = ro.r
    r.source(proj_root + "/R/src/writeGenusList.R")
    r.writeGenusList(proj_root)

def mergeUpdateAnnotations(skip_string=robj.NULL,proj_root="../.."):
    """
        Merge .csv annotations in trainset_tasks folder into one annotations.csv file in data folder
        :param str skip_string a string to look for in .csv file names then skip that file if found - used to exclude certain annotation tasks for example
    """
    r = ro.r
    r.source(proj_root + "/R/src/mergeUpdateAnnotations.R")
    r.mergeUpdateAnnotations(skip_string,proj_root)

def moveSelectImages(num_images, species,class_name,from_,to,excl_names):
    """
        Move a number of images from one folder to another based on some criteria

        :param int num_images: the number of images to move
        :param str species: move only images of this species
        :param str class_name: move only images of this annotation class i.e. dorsal
        :param str from_: the folder to move from
        :param str to: the folder to move to
        :param str excl_names: exclude specific image names i.e. 102030_1.jpg
    """
    r = ro.r
    r.source("../R/src/moveSelectImages.R")
    p = r.moveSelectImages(num_images, species, class_name,from_, to, excl_names)
    return p

def copyClassImagesForTraining(class_col,class_name, to, ntest, split_test_train=True, class_col2=robj.NULL,class_name2=robj.NULL,class_dir_override=robj.NULL,proj_dir="../.."):
    """
        Copies images of a specific annotation class to the "to" folder for training

        :param str class_col: the name of the column in which to filter for the class_name
        :param str class_name: the class name

        :param str to: the folder to move to (and create subdirs in)
        :param str ntest: the number of images to move to the test subdir (rest are placed in train subdir)

        :param str class_col2: a 2nd column in which to filter for a 2nd class_name
        :param str class_name2: the 2nd class name

        :param str class_dir_override: override the name of the created class folder to use the specified instead of the class_name
            - use to put multiple classes in the same folder for example
    """

    r = ro.r
    r.source("../../R/src/copyClassImagesForTraining.R")
    p = r.copyClassImagesForTraining(class_col=class_col,class_name=class_name, class_col2=class_col2,class_name2=class_name2,class_dir_override=class_dir_override, to=to, split_test_train=split_test_train, ntest=ntest, proj_dir=proj_dir)
    return p

def copyClassImagesToTrainingDirs(class_col,class_names,ntest,proj_dir="../.."):
    """
        More or less a wrapper for copyClassImagesForTraining, bringing it into the data oriented paradigm
    """
    to_dir = proj_dir + "/data/other/training_dirs/" + class_col
    print(to_dir)
    for c in class_names:
        print(c)
        copyClassImagesForTraining(class_col=class_col,class_name = c, proj_dir=proj_dir, to= to_dir + "/" + c, ntest=ntest, split_test_train=True, class_col2=robj.NULL,class_name2=robj.NULL,class_dir_override=robj.NULL)

def createTrainingTask(trainer, task_name,n,name_contains=robj.NULL,random_downloaded_only=False,proj_root="../.."):
    """
        Create a training task and move images to it

        :param str trainer: the name of the trainer
        :param str task_name: the name of the task, usually *N_*Details_*Date
        :param int n: the number of images to put in the training task
        :param int random_downloaded_only: whether to use randomly downloaded singles only
    """
    r = ro.r
    r.source("../../R/src/createEndTrainingTasks.R")
    r.createTrainingTask(trainer,task_name,n,name_contains,random_downloaded_only,proj_root)

def endTrainingTask(trainer, task_name, proj_root="../.."):
    """
        End a training task by deleting images in the folder and merging its annotations into data/annotations.csv

        :param str trainer: the name of the trainer
        :param str task_name: the name of the task, usually *N_*Details_*Date
    """
    r = ro.r
    r.source("../../R/src/createEndTrainingTasks.R")
    r.endTrainingTask(trainer,task_name, proj_root)

def getFilterImageIDs(annotation_field=robj.NULL,annotation_value=robj.NULL,
                      exclude_training=False,only_training=False,
                      exclude_segmented=False,only_segmented=False,
                      exclude_classified=False,only_classified=False,
                      image_ids_override=robj.NULL):
    """
        Convenience function to filter for image ids using various criteria
    """

    r = ro.r
    r.source("../../R/src/getFilterImageIDs.R")
    image_ids = r.getFilterImageIDs(annotation_field=annotation_field,annotation_value=annotation_value,
                                    exclude_training=exclude_training,only_training=only_training,
                                    exclude_segmented=exclude_segmented,only_segmented=only_segmented,
                                    exclude_classified=exclude_classified,only_classified=only_classified,
                                    image_ids_override=image_ids_override)
    return image_ids

def copyMaskImagesForTraining(from_="../data/segments/masks/train_masks", imgs_from="../data/all_images", to="", ntest=5):
    r = ro.r
    r.source("../../R/src/copyMaskImagesForTraining.R")
    r.copyMaskImagesForTraining(from_,imgs_from,to,ntest)