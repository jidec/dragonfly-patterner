
# merges all annotations in annotations folders into an annotations.csv file 
# held in data folder

mergeAnnotations <- function(){
  library(dplyr)
  library(plyr)
  csv_files <- dir(path= "../annotations", pattern='*.csv$', recursive = T)
  csv_files <- paste("../annotations/",csv_files)
  csv_files <- gsub(" ", "", csv_files)
  for(i in 1:length(csv_files)) {
    if(i == 1)
      df <- read.csv(csv_files[i])
    else
      df <- rbind.fill(df, read.csv(csv_files[i]))
  }
  df <- df[!duplicated(df$file),]
  
  segments <- list.files("../data/segments")
  segments <- cbind(segments, "TRUE")
  names(segments) <- c("file","has_segment")
  
  #todo - process when multiple shared annotations
  write.csv(df,file="../data/annotations.csv")
}