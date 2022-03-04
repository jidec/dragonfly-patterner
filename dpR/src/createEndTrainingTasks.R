# create one or more training tasks by making dirs and moving exclusive images 
createTrainingTasks <- function(trainers,task_name,from_,n)
{
  source("../dpR/src/moveSelectImages.R")
  annotations <- read.csv("../data/annotations.csv")
  used_names <- annotations$file
  
  for(i in 1:length(trainers))
  {
    target_dir <- paste0("../annotations/",trainers[i],"/",task_name)
    dir.create(target_dir)
    more_used_names <- moveSelectImages(num_images=200, from_="../data/random_images",
                     to = target_dir, excl_names=used_names)
    used_names <- c(used_names,more_used_names)
  }
}

# end a training task by 
endTrainingTasks <- function(trainers,task_name)
{
  source("../dpR/src/smallHelpers.R")
  source("../dpR/src/mergeAnnotations.R")
  for(i in 1:length(trainers))
  {
    target_dir <- paste0("../annotations/",trainers[i],"/",task_name)
    
    # delete all images from dir
    deleteImagesFromDir(target_dir)
    
    # merge all annotations including new ones from that dir
    mergeAnnotations()
  }
}