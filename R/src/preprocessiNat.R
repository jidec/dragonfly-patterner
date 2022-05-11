
# add useful cols and remove extraneous cols from iNat research grade data downloaded from GBIF 
preprocessiNat <- function(proj_root="../..")
{
  data <- read.csv(paste0(proj_root, "/data/inat_data.csv"),header=TRUE,row.names=NULL,sep="\t")
  
  # add column for number of images
  library(stringr)
  data$numImages <- str_count(data$mediaType, "StillImage")
  data$numImages[data$numImages == 0] <- 1
  print("Added numImages column containing number of images per observation")

  keeps <- c("occurrenceID","family","genus","species","infraspecificEpithet",
             "decimalLatitude","decimalLongitude","coordinateUncertaintyInMeters",
             "eventDate","day","month","year","taxonKey",
             "catalogNumber","identifiedBy","recordedBy","issue","numImages")
  data <- data[keeps]
  
  print("Removed extraneous columns")
  write.csv(data,paste0(proj_root, "/data/inat_data.csv"))
  print("Wrote new data .csv to data folder")
}