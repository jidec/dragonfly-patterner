
mergeImageObsData() <- function()
{
  # create by taking image names in all_images, merging in annotations and classifications
  list.files("../data/all_images",recursive = TRUE)
  
  image_obs <- data.frame(matrix(ncol = 3, nrow = 0))
  
  # annotations csv should have imageID is_segment_train class_dorsal_lateral_bad and class_is_perfect
  # inference csv should have imageID has_segment class_inferred 
  x <- c("image_id","is_segment_train", "is_classify_train","has_infer_segment","has_infer_classes",
         "class_dorsal_lateral_bad","class_is_perfect")
  colnames(df) <- x
     
  match("image_obs", list.files("../data"))
}