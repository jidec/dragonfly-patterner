# use to move images from the downloader folder to the raw_images folder
# use to move images from the all_images folder to training/test folders
# use to move images from the annotations folder to training/test folders or all_images
# use to move images from the all_images folder to annotations folder 

from <- "../downloaders/helpers/genus_image_records/iNat_images-Stylurus-raw_images"
num_images <- 1
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
  imgs_from
  imgs_from <- gsub(" ", "", imgs_from, fixed = TRUE)
  imgs_to <- paste(to, "/", imgs)
  imgs_to <- gsub(" ", "", imgs_to, fixed = TRUE)
  imgs_to
  # copy files 
  file.copy(from = imgs_from,
            to = imgs_to)
}

moveSelectImages(num_images=150, from="../downloaders/helpers/genus_image_records/iNat_images-Stylurus-raw_images",
                 to = "../data/all_images")

moveSelectImages(num_images = 150, from="../data/all_images",
                 to = "../annotations/Jacob/")