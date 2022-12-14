visualizeID <- function(id){
  library(imager)
  img <- load.image(paste0('E:/dragonfly-patterner/data/all_images/',id,".jpg"))
  return(plot(img))
}
visualizeID("INAT-49781305-4")