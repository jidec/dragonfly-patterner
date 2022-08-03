# view an image specifying an image name 
viewImageFromName <- function(name)
{
  library(imager)
  img <- load.image(paste0("../data/all_images/",name))
  plot(img)
}

# delete all files ending in .jpg from a directory
deleteImagesFromDir <- function(dir)
{
  files <- list.files(dir)
  files <- files[grepl(".jpg", files, fixed = TRUE)]
  files <- paste0(dir,"/",files)
  file.remove(files) 
}

# return the number of duplicate images between two folders 
findNDuplicates <- function(dir1, dir2, dir1_names_override=NULL)
{
  dir1files <- list.files(dir1)
  dir2files <- list.files(dir2)
  if(!is.null(dir1_names_override)){
    dir1files <- dir1_names_override
  }
  return(sum(duplicated(c(dir1files,dir2files))))
}

# save the names of images in a dir to keep track
recordImageNames <- function(dir,name)
{
  write.csv(list.files(dir), name)
}
