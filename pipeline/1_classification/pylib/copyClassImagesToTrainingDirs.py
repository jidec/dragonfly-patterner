def copyClassImagesToTrainingDirs(training_dir_name, class_col,class_names,ntest,proj_dir="../.."):
    """
        More or less a wrapper for copyClassImagesForTraining, bringing it into the data oriented paradigm
    """
    # expand class cols to work with getFilterImageIDs
    for c in class_names:
        # ids = getFilterImageIDs(fields=class_col, values=c)
        #source_files = proj_dir + "/data/all_images/" + ids
        # create training dir name folder
        # dest_files = proj_dir + "/data/other/temp_training_dirs/" + training_dir_name + "/" + ids
        # shutil.copy

