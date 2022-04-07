
# add useful cols and remove extraneous cols from iNat research grade data downloaded from GBIF 
preprocessiNat() <- function()
{
  data <- read.csv("../data/inat_odonata_usa.csv",header=TRUE,row.names=NULL,sep="\t")
  
  # add column for number of images
  library(stringr)
  data$numImages <- str_count(data$mediaType, "StillImage")
  data$numImages[data$numImages == 0] <- 1
  
  keeps <- c("occurrenceID","family","genus","species","infraspecificEpithet",
             "decimalLatitude","decimalLongitude","coordinateUncertaintyInMeters",
             "eventDate","day","month","year","taxonKey",
             "catalogNumber","identifiedBy","recordedBy","issue","numImages")
  data <- data[keeps]
  
  write.csv(data,"../data/inat_odonata_usa.csv")
}