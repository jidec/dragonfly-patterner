# use to move images from the downloader folder to the raw_images folder
# use to move images from the all_images folder to training/test folders
# use to move images from the annotations folder to training/test folders or all_images
# use to move images from the all_images folder to annotations folder 

#num_images = 6
#from="../data/all_images"
#to = "../annotations/Louis/2-4-22_60Segments"
#excl_names = moved
#class_name <- "dorsal"

moveSelectImages <- function(num_images= NULL, species = NULL, class_name = NULL, from_, to, excl_names= NULL)
{
  from <- from_
  # read in all Odonata observations and annotations
  data <- read.csv("../../data/inat_odonata_usa.csv",header=TRUE,row.names=NULL,sep="\t")
  annotations <- read.csv("../../data/annotations.csv")
  
  if(!is.null(species)){ # if supplied a species, get only that species
    data <- subset(data, scientificName = species)
  }
  inat_ids <- data$catalogNumber # get inat ids from data
  # TODO - match inat_ids to imgs
  
  # get all images in from directory 
  imgs <- list.files(from) 
  
  # get files of class from annotations
  if(!is.null(class_name)){
    class_files <- annotations[annotations$dorsal_lateral==class_name,]$file
    imgs <- imgs[imgs %in% class_files]
  }

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

# used to copy from task folders to data folder
copyMasks <- function(from_, to="../data/segments/masks"){
  
  from <- from_
  imgs <- list.files(from) 
  
  # get masks
  imgs <- imgs[grepl("mask", imgs, fixed = TRUE)]

  # get from and fix
  imgs_from <- paste0(from, "/", imgs)
  imgs_to <- paste0(to, "/", imgs)
  
  # copy files
  file.copy(from = imgs_from, to = imgs_to)
}