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

findNDuplicates <- function(dir1, dir2)
{
  dir1files <- list.files(dir1)
  dir2files <- list.files(dir2)
  return(sum(duplicated(c(dir1files,dir2files))))
}