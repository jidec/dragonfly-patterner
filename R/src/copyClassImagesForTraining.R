#class_col = "class"
#class_name = "dorsal"
#from_ = "../data/random_images"
#to = "../experiments/odo_view_classifier/odo_view_data"
#ntest=10

# copies images of a specific class to the target dir in a new folder named with the class name,or the class_dir_override name
copyClassImagesForTraining <- function(class_col, class_name, class_col2=NULL, class_name2=NULL, class_dir_override=NULL, to, split_test_train=FALSE, ntest=10){
  
  library(rlang)
  
  # enforced to draw images from all_images folder
  from <- "../../data/all_images"
  
  # load annotations
  annotations <- read.csv("../../data/annotations.csv")
  
  imgs <- list.files(from) 
  
  # get image names that are annotated
  names <- dplyr::filter(annotations, !!sym(class_col) == class_name)
  if(!is.null(class_col2)){
    names <- dplyr::filter(annotations, !!sym(class_col2) == class_name2)
  }
  
  names <- paste0(names$imageID, ".jpg")
  
  # get matching images from directory
  class_imgs <- imgs[imgs %in% names]
  #class_imgs <- imgs[match(imgs,names)]
  
  # create to dir if it doesn't exist
  if(!file.exists(to)){
    dir.create(to)
  }
  
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
    class_imgs_from_test
    class_imgs_from_train <- class_imgs_from[ntest+1:length(class_imgs_from)]
    
    # override class dir if included (use to send multiple classes to the same folder) 
    if(!is.null(class_dir_override)){
      # get to and fix 
      class_imgs_to_train <- paste0(to, "/", "train/", class_dir_override, "/", class_imgs)
      class_imgs_to_test <- paste0(to, "/", "test/", class_dir_override, "/", class_imgs)
      class_imgs_to_test <- class_imgs_to_test[1:ntest]
    }
    
    else{
      # get to and fix 
      class_imgs_to_train <- paste0(to, "/", "train/", class_name, "/", class_imgs)
      class_imgs_to_test <- paste0(to, "/", "test/", class_name, "/", class_imgs)
      class_imgs_to_test <- class_imgs_to_test[1:ntest]
    }
    
    
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