
# copy masks and mask images to a training folder 
copyMaskImagesForTraining <- function(from_="../data/segments/masks/train_masks", 
                                      imgs_from="../data/all_images", to="", ntest=5){
  from <- from_
  # get mask files from masks only folder
  masks <- list.files(from) 
  
  # subset test and train masks
  masks_test <- masks[1:ntest]
  masks_train <- masks[ntest+1:length(masks)]
  
  imgs_test <- gsub("_mask","", masks_test, fixed = TRUE)
  imgs_train <- gsub("_mask","", masks_train, fixed = TRUE)
  
  # create train test img and mask dirs
  dir.create(paste0(to, "/train"))
  dir.create(paste0(to, "/val"))
  dir.create(paste0(to, "/train/Image"))
  dir.create(paste0(to, "/train/Mask"))
  dir.create(paste0(to, "/val/Image"))
  dir.create(paste0(to, "/val/Mask"))
  
  
  # get mask from and to 
  masks_from_test <- paste0(from, "/", masks_test)
  masks_to_test <- paste0(to, "/val/Mask/", masks_test)
  masks_from_train <- paste0(from, "/", masks_train)
  masks_to_train <- paste0(to, "/train/Mask/", masks_train)
  
  # get mask from and to 
  imgs_from_test <- paste0(imgs_from, "/", imgs_test)
  
  #list.dirs(imgs_from, recursive=TRUE)
  
  imgs_to_test <- paste0(to, "/val/Image/", imgs_test)
  imgs_from_train <- paste0(imgs_from, "/", imgs_train)
  imgs_to_train <- paste0(to, "/train/Image/", imgs_train)
  
  # copy files
  file.copy(from = masks_from_test, to = masks_to_test)
  file.copy(from = masks_from_train, to = masks_to_train)
  file.copy(from = imgs_from_test, to = imgs_to_test)
  file.copy(from = imgs_from_train, to = imgs_to_train)
}
