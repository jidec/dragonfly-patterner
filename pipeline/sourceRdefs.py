import rpy2.robjects as ro

def mergeAnnotations():
    r = ro.r
    r.source("../dpR/src/mergeAnnotations.R")
    r.mergeAnnotations()

def moveSelectImages(num_images, species,class_name,from_,to,excl_names):
    r = ro.r
    r.source("../dpR/src/moveSelectImages.R")
    p = r.moveSelectImages(num_images, species, class_name,from_, to, excl_names)
    return p

def moveAnnotationClassImages(class_col,class_name,class_col2,class_name2,class_dir_override, from_, to, split_test_train, ntest):
    r = ro.r
    r.source("../dpR/src/moveSelectImages.R")
    p = r.moveAnnotationClassImages(class_col,class_name, class_col2,class_name2,class_dir_override,from_, to, split_test_train, ntest)
    return p

def createTrainingTasks(trainers, task_name, from_,n):
    r = ro.r
    r.source("../dpR/src/createEndTrainingTasks.R")
    r.createTrainingTasks(trainers,task_name,from_,n)

def endTrainingTasks(trainers, task_name):
    r = ro.r
    r.source("../dpR/src/createEndTrainingTasks.R")
    r.endTrainingTasks(trainers,task_name)