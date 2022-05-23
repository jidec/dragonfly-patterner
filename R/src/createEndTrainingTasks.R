 #trainers = "Rob"
#task_name = "test"
#n=100
#py=TRUE

# create one or more training tasks by making dirs and moving exclusive images
createTrainingTask <- function(trainers,task_name,n,name_contains,random_downloaded_only=FALSE,proj_root="../..")
{
  library(stringr)
  
  source("../../R/src/moveSelectImages.R")
  
  print(paste0("Creating training task ", trainers[1],"/",task_name,"..."))
  
  print("Reading in annotations")
  
  # read in annotations and get names of images already in training set
  annotations <- read.csv(paste0(proj_root, "/data/annotations.csv"),row.names = 1)
  used_names <- paste0(annotations$imageID, ".jpg")

  # if only use random downloads, add names to underscores (non random dls) to used names
  if(random_downloaded_only){
    names <- list.files(paste0(proj_root, "/data/all_images"))
    names <- names[str_detect(names,"INAT-")]
    used_names <- c(used_names,names)
    names <- names[str_detect(names,"OC-")]
    used_names <- c(used_names,names)
  }

  if(!is.null(name_contains)){
    print("Filtering for names containing specified string")
    #proj_root = ".."
    #name_contains = "-h"
    names <- list.files(paste0(proj_root, "/data/all_images"))
    names <- names[str_detect(names,name_contains,negate=TRUE)]
    used_names <- c(used_names,names)
  }
  for(i in 1:length(trainers))
  {
    print("Creating new training dir")
    #i = 1
    target_dir <- paste0(proj_root, "/trainset_tasks/",trainers[i],"/",task_name)
    dir.create(target_dir)
    print("Moving images to new dir")
    print(used_names)
    more_used_names <- moveSelectImages(num_images=n, from_=paste0(proj_root,"/data/all_images"),
                     to = target_dir, excl_names=used_names)
    more_used_names <- str_remove(more_used_names,".jpg")
    more_used_names <- as.data.frame(more_used_names)
    more_used_names$in_task <- rep(TRUE,nrow(more_used_names))
   
    colnames(more_used_names) <- c("imageID","in_task")
    print("Adding names to annotations")
    annotations <- merge(annotations, more_used_names, by="imageID",all=TRUE)
    write.csv(annotations,paste0(proj_root, "/data/annotations.csv"))
  }
  
  #names <- paste0(annotations$imageID, ".jpg")
  # verify exclusive
  #findNDuplicates(dir1="../trainset_tasks/Rob/test1",dir2="../trainset_tasks/Rob/test2")
}

# end a training task by deleting images from dir and merging 
endTrainingTask <- function(trainers,task_name,proj_root="../..")
{
  source(paste0(proj_root, "/R/src/smallHelpers.R"))
  source(paste0(proj_root, "/R/src/mergeUpdateAnnotations.R"))
  for(i in 1:length(trainers))
  {
    target_dir <- paste0(proj_root, "/trainset_tasks/",trainers[i],"/",task_name)
    
    img_files <- list.files(target_dir)
    
    # get all masks from folder and copy to data folder
    mask_files <- img_files[grepl("_mask", img_files, fixed = TRUE)]
    masks_from <- paste0(target_dir,mask_files)
    masks_to <- paste0(proj_root, "/data/segments/masks/train_masks")
    file.copy(from = masks_from, to = masks_to)
    
    # delete all images from dir
    deleteImagesFromDir(target_dir)
    
    # merge all annotations including new ones from that dir
    mergeUpdateAnnotations(proj_root=proj_root)
  }
}