
#from = "../annotations/Jacob/2-4-22_60Segments"

# write a csv of the names of images that were segmented 
recordSegmentationTask <- function(from)
{
  imgs <- list.files(from) 
  imgs <- imgs[grepl("_mask", imgs)] # get mask names rather than image names
  write.csv(imgs, paste0(from,"/session.csv"))
}
