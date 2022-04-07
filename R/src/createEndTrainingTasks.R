 #trainers = "Rob"
#task_name = "test"
#n=100
#py=TRUE

# create one or more training tasks by making dirs and moving exclusive images
createTrainingTask <- function(trainers,task_name,n)
{
  library(stringr)
  
  source("../../R/src/moveSelectImages.R")
    
  # read in annotations and get names of images already in training set
  annotations <- read.csv("../../data/annotations.csv",row.names = "X")
  used_names <- paste0(annotations$imageID, ".jpg")
  
  for(i in 1:length(trainers))
  {
    #i = 1
    target_dir <- paste0("../../trainset_tasks/",trainers[i],"/",task_name)
    dir.create(target_dir)
    more_used_names <- moveSelectImages(num_images=n, from_="../../data/all_images",
                     to = target_dir, excl_names=used_names)
    more_used_names <- str_remove(more_used_names,".jpg")
    more_used_names <- as.data.frame(more_used_names)
    more_used_names$in_task <- rep(TRUE,nrow(more_used_names))
   
    colnames(more_used_names) <- c("imageID","in_task")
    annotations <- merge(annotations, more_used_names, by="imageID",all=TRUE)
    write.csv(annotations,"../../data/annotations.csv")
  }
  
  #names <- paste0(annotations$imageID, ".jpg")
  # verify exclusive
  #findNDuplicates(dir1="../trainset_tasks/Rob/test1",dir2="../trainset_tasks/Rob/test2")
}

# end a training task by deleting images from dir and merging 
endTrainingTask <- function(trainers,task_name)
{
  source("../../R/src/smallHelpers.R")
  source("../../R/src/mergeUpdateAnnotations.R")
  for(i in 1:length(trainers))
  {
    target_dir <- paste0("../../trainset_tasks/",trainers[i],"/",task_name)
    
    img_files <- list.files(target_dir)
    
    # get all masks from folder and copy to data folder
    mask_files <- mask_files[grepl("_mask", img_files, fixed = TRUE)]
    masks_from <- paste0(target_dir,mask_files)
    masks_to <- paste0("../../data/segments/masks/train_masks")
    file.copy(from = masks_from, to = masks_to)
    
    # delete all images from dir
    deleteImagesFromDir(target_dir)
    
    # merge all annotations including new ones from that dir
    mergeUpdateAnnotations()
  }
}