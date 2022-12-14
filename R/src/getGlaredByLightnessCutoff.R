getGlaredByLightnessCutoff <- function(image_dir, lightness_cutoff=0.9, percent_past_cutoff=0.1){
  
  print("Loading images...")
  library(imager)
  images <- load.dir(image_dir)
  
  l_channels <- list()
  
  print("Checking if glared")
  is_glared_vect <- c()
  for(img in images){
    l <- (img[,,,1] + img[,,,2] + img[,,,3]) / 3
    is_glared <- (sum(l > lightness_cutoff) / length(l)) > percent_past_cutoff
    is_glared_vect <- c(is_glared_vect,is_glared)
  }
  
  names(is_glared_vect) <- list.files(image_dir)
  
  return(is_glared_vect)
}