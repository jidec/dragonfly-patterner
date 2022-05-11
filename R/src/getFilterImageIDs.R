
# used to select images for inference
getFilterImageIDs <- function(annotation_field=NULL,annotation_value=NULL,
                              exclude_training=FALSE,only_training=FALSE,
                              exclude_segmented=FALSE,only_segmented=FALSE,
                              exclude_classified=FALSE, only_classified=FALSE,
                              image_ids_override=NULL,
                              img_dir="../../data/all_images",annot_dir="../../data/annotations.csv")
{
  library(dplyr)
  library(rlang)
  library(stringr)
  
  # read in image_ids
  image_ids <- str_remove(list.files(img_dir),".jpg")
  
  # set starting image ids to override
  if(!is.null(image_ids_override)){
    image_ids <- image_ids_override
  }
  
  # filter 
  annotations <- read.csv(annot_dir)
  if(exclude_training){image_ids <- setdiff(image_ids,annotations$imageID)}
  if(only_training){image_ids <- intersect(image_ids,annotations$imageID)}
  
  segmented_ids <- dplyr::filter(annotations, has_segment == TRUE)$imageID
  if(exclude_segmented){image_ids <- setdiff(image_ids, segmented_ids)}
  if(only_segmented){image_ids <- intersect(image_ids,segmented_ids)}
  
  classified_ids <- dplyr::filter(annotations, is.na(dorsal_lateral_dorsolateral_bad))$imageID
  if(exclude_classified){image_ids <- setdiff(image_ids, classified_ids)}
  if(only_classified){image_ids <- intersect(image_ids,classified_ids)}
  
  # filter for field
  if(!is.null(annotation_field)){
    filter_ids <- dplyr::filter(annotations, !!sym(annotation_field) != annotation_value)
    image_ids <- setdiff(image_ids, filter_ids)
  }
  
  return(image_ids)
}