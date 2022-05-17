
# add useful cols and remove extraneous cols from data downloaded from GBIF (including iNat research grade, AntWeb, or others)
preprocessGBIF <- function(gbif_csv_name,proj_root="../..")
{
  data <- read.csv(paste0(proj_root, "/data/", gbif_csv_name, ".csv"),header=TRUE,row.names=NULL,sep="\t",quote="")
  
  # add column for number of images
  library(stringr)
  data$numImages <- str_count(data$mediaType, "StillImage")
  # data$numImages[data$numImages == 0] <- 1
  print("Added numImages column containing number of images per observation")

  keeps <- c("occurrenceID","family","genus","species","infraspecificEpithet",
             "decimalLatitude","decimalLongitude","coordinateUncertaintyInMeters",
             "eventDate","day","month","year","taxonKey",
             "catalogNumber","identifiedBy","recordedBy","issue","numImages")
  data <- data[keeps]
  
  print("Removed extraneous columns")
  
  data <- data[data$numImages != 0,]
  print("Removed observations without images")
  
  write.csv(data,paste0(proj_root, "/data/", gbif_csv_name, ".csv"))
  print("Wrote updated data .csv to data folder")
}
