
# merges all annotations in annotations folders into an annotations.csv file 
# held in data folder
mergeUpdateAnnotations <- function(){
  library(dplyr)
  library(plyr)
  library(stringr)
  
  # merge all files using rbind.fill
  csv_files <- dir(path= "../../trainset_tasks", pattern='*.csv$', recursive = T)
  csv_files <- paste("../../trainset_tasks/",csv_files)
  csv_files <- gsub(" ", "", csv_files)
  for(i in 1:length(csv_files)) {
    if(i == 1)
      df <- read.csv(csv_files[i])
    else
      df <- rbind.fill(df, read.csv(csv_files[i]))
  }
  df <- df[!duplicated(df$file),]
  
  # get image names with training masks and merge with df 
  segments <- list.files("../../data/segments/masks/train_masks")
  segments <- str_replace(segments,"_mask","")
  segments <- cbind(segments, "TRUE")
  colnames(segments) <- c("file","has_segment")
  df <- merge(segments, df, by = 'file',all=TRUE)
  
  # join differently named dorsal_lateral_bad classifications
  joined_dlb <- paste0(df$dorsal_lateral, df$class)
  joined_dlb <- str_replace(joined_dlb,"NA","")
  joined_dlb <- str_replace(joined_dlb,"0","bad")
  df$dorsal_lateral_bad <- joined_dlb

  keeps <- c("file","has_segment","dorsal_lateral_bad","is_perfect")
  df <- df[keeps]
  
  colnames(df) <- c("imageID","has_segment","dorsal_lateral_bad","is_perfect")
  
  # fix cols, replacing weird values
  df$imageID <- str_replace(df$imageID,".jpg","")
  df$has_segment[is.na(df$has_segment)] <- FALSE
  
  df$is_perfect <- str_replace(df$is_perfect,"bad","FALSE")
  df$is_perfect <- str_replace(df$is_perfect,"0","FALSE")
  df$is_perfect <- str_replace(df$is_perfect,"1","TRUE")
  
  # example getting imageIDs with a criterion
  #df[df$has_segment == TRUE,]$imageID
  
  #todo - process when multiple shared annotations
  write.csv(df,file="../../data/annotations.csv")
}
