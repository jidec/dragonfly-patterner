import rpy2.robjects as ro
import os
import rpy2.robjects as robj

# currently sourcing must be done from the relative directory of scripts in pipeline folders

def preprocessiNat():
    r = ro.r
    r.source("../../R/src/preprocessiNat.R")
    r.preprocessiNat()

def writeGenusList():
    r = ro.r
    r.source("../../R/src/writeGenusList.R")
    r.writeGenusList()

def mergeUpdateAnnotations():
    r = ro.r
    r.source("../../R/src/mergeUpdateAnnotations.R")
    r.mergeUpdateAnnotations()

def moveSelectImages(num_images, species,class_name,from_,to,excl_names):
    r = ro.r
    r.source("../R/src/moveSelectImages.R")
    p = r.moveSelectImages(num_images, species, class_name,from_, to, excl_names)
    return p

def copyClassImagesForTraining(class_col,class_name, to, ntest, split_test_train=True, class_col2=robj.NULL,class_name2=robj.NULL,class_dir_override=robj.NULL):
    r = ro.r
    r.source("../../R/src/copyClassImagesForTraining.R")
    p = r.copyClassImagesForTraining(class_col,class_name, class_col2,class_name2,class_dir_override, to, split_test_train, ntest)
    return p

def createTrainingTask(trainer, task_name,n):
    r = ro.r
    r.source("../../R/src/createEndTrainingTasks.R")
    r.createTrainingTask(trainer,task_name,n)

def endTrainingTask(trainer, task_name):
    r = ro.r
    r.source("../../R/src/createEndTrainingTasks.R")
    r.endTrainingTask(trainer,task_name)

def getFilterImageIDs(annotation_field=robj.NULL,annotation_value=robj.NULL,
                      exclude_training=False,only_training=False,
                      exclude_segmented=False,only_segmented=False,
                      exclude_classified=False,only_classified=False,
                      image_ids_override=robj.NULL):
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