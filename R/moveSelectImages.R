# use to move images from the downloader folder to the raw_images folder
# use to move images from the all_images folder to training/test folders
# use to move images from the annotations folder to training/test folders or all_images
# use to move images from the all_images folder to annotations folder 

moveSelectImages <- function(num_images= NULL, species = NULL, from, to)
{
  # read in all Odonata observations
  data <- read.csv("../data/inat_odonata_usa.csv",header=TRUE,row.names=NULL,sep="\t")
  
  if(!is.null(species)){ # if supplied a species, get only that species
    data <- subset(data, scientificName = species)
  }
  inat_ids <- data$catalogNumber # get inat ids from data
  # TODO - match inat_ids to imgs
  
  # get all images in from directory 
  imgs <- list.files(from) 
  
  # shuffle images
  imgs <- sample(imgs)
  # if num_images supplied get random images, otherwise just use all 
  if(!is.null(num_images)){ 
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
}

moveAnnotationClassImages <- function(class_name, from, to){
  
  annotations <- read.csv("../data/annotations.csv")
  
  imgs <- list.files(from) 
  
  # get image names that are annotated
  names <- dplyr::filter(annotations, dorsal_lateral == class_name)
  names <- names$file
  
  # get matching images from directory
  class_imgs <- imgs[match(imgs,names)]

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