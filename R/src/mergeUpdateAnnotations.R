
# root = "../pipeline/1_classification/"
# merges all annotations in annotations folders into an annotations.csv file 
# held in data folder
mergeUpdateAnnotations <- function(skip_string=NULL,proj_root="../.."){
  library(dplyr)
  library(plyr)
  library(stringr)

  # gather all csv files in trainset_tasks
  csv_files <- dir(path= paste0(proj_root, "/trainset_tasks"), pattern='*.csv$', recursive = T)
  csv_files <- paste0(proj_root,"/trainset_tasks/",csv_files)
  print(csv_files)
  
  # if a skip string provided, skip files containing that string 
  if(!is.null(skip_string)){
    csv_files <- csv_files[str_detect(csv_files,skip_string,negate = TRUE)]
  }
  
  # for each csv file, merge and remove duplicates
  for(i in 1:length(csv_files)) {
    if(i == 1)
      df <- read.csv(csv_files[i],row.names = 1)
    else
      df <- rbind.fill(df, read.csv(csv_files[i]))
  }
  df <- df[!duplicated(df$file),]
  
  # replace file col with imageID col
  colnames(df)[1] <- "imageID"
  df$imageID <- str_replace(df$file,".jpg","")
  
  # get image names with training masks and merge with df 
  # segments <- list.files(paste0(proj_root,"/data/masks/train_masks"))
  # if(length(segments) > 0){
  #  segments <- str_replace(segments,"_mask","")
  #  segments <- cbind(segments, "TRUE")
  #  colnames(segments) <- c("file","has_segment")
  #  print(df)
  #  df <- merge(segments, df, by = 'file',all=TRUE)
  #  print(df)
  #  df$has_segment[is.na(df$has_segment)] <- FALSE
  #}

  #keeps <- c("file","has_segment","dorsal_lateral_dorsolateral_bad","is_perfect")
  #df <- df[keeps]
  
  #colnames(df) <- c("imageID","has_segment","dorsal_lateral_dorsolateral_bad","is_perfect")
  
  # fix cols, replacing weird values
  # df$imageID <- str_replace(df$imageID,".jpg","")
  
  # example getting imageIDs with a criterion
  #df[df$has_segment == TRUE,]$imageID
  
  #todo - process when multiple shared annotations
  write.csv(df,file=paste0(proj_root,"/data/annotations.csv"))
  print("Wrote merged annotations to data/annotations.csv")
}
