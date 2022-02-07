# use to move images from the downloader folder to the raw_images folder
# use to move images from the all_images folder to training/test folders
# use to move images from the annotations folder to training/test folders or all_images
# use to move images from the all_images folder to annotations folder 

num_images = 6
from="../data/all_images"
to = "../annotations/Louis/2-4-22_60Segments"
excl_names = moved
class_name <- "dorsal"
  
moveSelectImages <- function(num_images= NULL, species = NULL, class_name = NULL, from, to, excl_names= NULL)
{
  # read in all Odonata observations and annotations
  data <- read.csv("../data/inat_odonata_usa.csv",header=TRUE,row.names=NULL,sep="\t")
  annotations <- read.csv("../data/annotations.csv")
  
  if(!is.null(species)){ # if supplied a species, get only that species
    data <- subset(data, scientificName = species)
  }
  inat_ids <- data$catalogNumber # get inat ids from data
  # TODO - match inat_ids to imgs
  
  # get all images in from directory 
  imgs <- list.files(from) 
  
  # get files of class from annotations
  class_files <- annotations[annotations$dorsal_lateral==class_name,]$file
  imgs <- imgs[imgs %in% class_files]

  # exclude excl_names
  if(!is.null(excl_names)){
    imgs <- imgs[!(imgs %in% excl_names)]
  }
  
  # shuffle images
  imgs <- sample(imgs)

  # if num_images supplied get random images, otherwise just use all 
  if(!is.null(num_images) && num_images < length(imgs)){ 
    imgs <- sample(imgs,num_images)
  }
  # prep to and from char vectors
  imgs_from <- paste(from, "/", imgs)
  imgs_from <- gsub(" ", "", imgs_from, fixed = TRUE)
  imgs_to <- paste(to, "/", imgs)
  imgs_to <- gsub(" ", "", imgs_to, fixed = TRUE)
    
  # copy files 
  file.copy(from = imgs_from,
            to = imgs_to)
  return(imgs)
}

#class_name = "lateral"
#from = "../data/all_images"
#to = "../experiments/odo_view_classifier/odo_view_data"

moveAnnotationClassImages <- function(class_name, from, to, split_test_train=FALSE, ntest=10){
  
  annotations <- read.csv("../data/annotations.csv")
  
  annotations$file == "103224054_798.jpg"
  annotations[annotations$file == "103224054_798.jpg", ]
  imgs <- list.files(from) 

  # get image names that are annotated
  names <- dplyr::filter(annotations, dorsal_lateral == class_name)
  names <- names$file
  
  # get matching images from directory
  class_imgs <- imgs[imgs %in% names]
  #class_imgs <- imgs[match(imgs,names)]
  
  if(!split_test_train)
  {
    # get from and fix
    class_imgs_from <- paste(from, "/", class_imgs)
    class_imgs_from <- gsub(" ", "", class_imgs_from, fixed = TRUE)
    
    # get to and fix 
    class_imgs_to <- paste(to, "/", class_name, "/", class_imgs)
    class_imgs_to <- gsub(" ", "", class_imgs_to, fixed = TRUE)
    
    # create class dir
    d <- gsub(" ", "", paste(to, "/", class_name), fixed = TRUE)
    dir.create(d)
    
    # copy files
    file.copy(from = class_imgs_from, to = class_imgs_to)
  }
  
  else
  {
    # create train test dirs
    d <- gsub(" ", "", paste(to, "/train"), fixed = TRUE)
    dir.create(d)
    d <- gsub(" ", "", paste(to, "/test"), fixed = TRUE)
    dir.create(d)
    
    # get from and fix
    class_imgs_from <- paste(from, "/", class_imgs)
    class_imgs_from <- gsub(" ", "", class_imgs_from, fixed = TRUE)
    class_imgs_from_test <- class_imgs_from[1:ntest]
    class_imgs_from_train <- class_imgs_from[ntest+1:length(class_imgs_from)]
    
    # get to and fix 
    class_imgs_to_train <- paste(to, "/", "train/", class_name, "/", class_imgs)
    class_imgs_to_train <- gsub(" ", "", class_imgs_to_train, fixed = TRUE)
    class_imgs_to_test <- paste(to, "/", "test/", class_name, "/", class_imgs)
    class_imgs_to_test <- gsub(" ", "", class_imgs_to_test, fixed = TRUE)
    
    # create class dirs
    d <- gsub(" ", "", paste(to, "/train/", class_name), fixed = TRUE)
    dir.create(d)
    d <- gsub(" ", "", paste(to, "/test/", class_name), fixed = TRUE)
    dir.create(d)
    
    # copy files
    file.copy(from = class_imgs_from_train, to = class_imgs_to_train)
    file.copy(from = class_imgs_from_test, to = class_imgs_to_test)
  }
}